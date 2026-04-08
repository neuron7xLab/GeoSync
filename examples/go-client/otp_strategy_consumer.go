// OTP Strategy Consumer — integrates CoherenceBridge regime signals
// into Ali Askar's Open Trading Platform strategy framework.
//
// Architecture fit:
//   OTP uses Kafka for all event distribution (orders, trades, market-data).
//   This consumer subscribes to coherence.signals.v1 and provides:
//     1. Thread-safe GetCurrentSignal(instrument) for strategy engines
//     2. Position blocking when regime = CRITICAL or DECOHERENT
//     3. Signal passthrough when regime = METASTABLE && risk_scalar > 0.7
//
// OTP service paths that consume this:
//   go/execution-venues/order-router  — SmartRouter: block in CRITICAL
//   go/vwap-strategy                  — scale slices by risk_scalar
//   go/order-monitor                  — expose regime to React UI
//
// Wire into OTP:
//   In your strategy's main(), add:
//     consumer := NewRegimeConsumer(kafkaBrokers, "coherence.signals.v1")
//     go consumer.Run(ctx)
//     // In strategy loop:
//     sig := consumer.GetCurrentSignal("EURUSD")
//     if sig != nil && consumer.ShouldBlockNewPositions("EURUSD") {
//         return // regime not favorable
//     }
//     positionSize *= sig.RiskScalar
//
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"

	"github.com/IBM/sarama"
	cbv1 "github.com/neuron7xLab/coherence-bridge/proto/v1"
	"google.golang.org/protobuf/proto"
)

// RegimeConsumer reads CoherenceBridge signals from Kafka and provides
// thread-safe access for OTP strategy engines.
type RegimeConsumer struct {
	brokers string
	topic   string

	mu       sync.RWMutex
	signals  map[string]*cbv1.RegimeSignal // instrument → latest signal
	seqNums  map[string]uint64             // instrument → last sequence_number
	gapCount map[string]uint64             // instrument → gap counter
}

// NewRegimeConsumer creates a consumer for the given Kafka cluster and topic.
func NewRegimeConsumer(brokers, topic string) *RegimeConsumer {
	return &RegimeConsumer{
		brokers:  brokers,
		topic:    topic,
		signals:  make(map[string]*cbv1.RegimeSignal),
		seqNums:  make(map[string]uint64),
		gapCount: make(map[string]uint64),
	}
}

// Run starts consuming from Kafka. Blocks until ctx is cancelled.
func (rc *RegimeConsumer) Run(ctx context.Context) error {
	config := sarama.NewConfig()
	config.Consumer.Return.Errors = true
	config.Consumer.Offsets.Initial = sarama.OffsetNewest

	consumer, err := sarama.NewConsumer([]string{rc.brokers}, config)
	if err != nil {
		return fmt.Errorf("kafka connect: %w", err)
	}
	defer consumer.Close()

	partConsumer, err := consumer.ConsumePartition(rc.topic, 0, sarama.OffsetNewest)
	if err != nil {
		return fmt.Errorf("kafka partition: %w", err)
	}
	defer partConsumer.Close()

	log.Printf("RegimeConsumer: listening on %s/%s", rc.brokers, rc.topic)

	for {
		select {
		case <-ctx.Done():
			return nil
		case msg := <-partConsumer.Messages():
			rc.handleMessage(msg)
		case err := <-partConsumer.Errors():
			log.Printf("RegimeConsumer kafka error: %v", err)
		}
	}
}

func (rc *RegimeConsumer) handleMessage(msg *sarama.ConsumerMessage) {
	sig := &cbv1.RegimeSignal{}
	if err := proto.Unmarshal(msg.Value, sig); err != nil {
		log.Printf("RegimeConsumer unmarshal error: %v", err)
		return
	}

	rc.mu.Lock()
	defer rc.mu.Unlock()

	inst := sig.Instrument

	// Gap detection via sequence_number
	if lastSeq, ok := rc.seqNums[inst]; ok {
		if sig.SequenceNumber > lastSeq+1 {
			gap := sig.SequenceNumber - lastSeq - 1
			rc.gapCount[inst] += gap
			log.Printf("RegimeConsumer: %s gap detected: %d missing signals (seq %d→%d)",
				inst, gap, lastSeq, sig.SequenceNumber)
		}
	}
	rc.seqNums[inst] = sig.SequenceNumber
	rc.signals[inst] = sig
}

// GetCurrentSignal returns the latest regime signal for an instrument.
// Thread-safe. Returns nil if no signal received yet.
func (rc *RegimeConsumer) GetCurrentSignal(instrument string) *cbv1.RegimeSignal {
	rc.mu.RLock()
	defer rc.mu.RUnlock()
	return rc.signals[instrument]
}

// ShouldBlockNewPositions returns true when the regime is not favorable
// for opening new positions. This is the primary risk gate.
//
// Blocking conditions:
//   - CRITICAL regime (herding/crash precursor, R→1)
//   - DECOHERENT regime (no signal edge, R→0)
//   - risk_scalar < 0.3 (gamma far from metastable)
//   - No signal available (fail-closed)
func (rc *RegimeConsumer) ShouldBlockNewPositions(instrument string) bool {
	rc.mu.RLock()
	defer rc.mu.RUnlock()

	sig, ok := rc.signals[instrument]
	if !ok || sig == nil {
		return true // fail-closed: no data = no trading
	}

	switch sig.Regime {
	case cbv1.RegimeType_REGIME_CRITICAL:
		return true
	case cbv1.RegimeType_REGIME_DECOHERENT:
		return true
	case cbv1.RegimeType_REGIME_UNKNOWN:
		return true
	}

	if sig.RiskScalar < 0.3 {
		return true
	}

	return false
}

// IsMetastableEdge returns true when conditions are optimal for trading:
// METASTABLE regime with sufficient risk_scalar and confidence.
func (rc *RegimeConsumer) IsMetastableEdge(instrument string) bool {
	rc.mu.RLock()
	defer rc.mu.RUnlock()

	sig, ok := rc.signals[instrument]
	if !ok || sig == nil {
		return false
	}

	return sig.Regime == cbv1.RegimeType_REGIME_METASTABLE &&
		sig.RiskScalar > 0.7 &&
		sig.RegimeConfidence > 0.6
}

// GetGapCount returns the number of detected sequence gaps for an instrument.
func (rc *RegimeConsumer) GetGapCount(instrument string) uint64 {
	rc.mu.RLock()
	defer rc.mu.RUnlock()
	return rc.gapCount[instrument]
}

// RiskScaledSize adjusts a base position size by the current risk_scalar.
// Returns 0 if positions should be blocked.
func (rc *RegimeConsumer) RiskScaledSize(instrument string, baseSize float64) float64 {
	if rc.ShouldBlockNewPositions(instrument) {
		return 0
	}
	sig := rc.GetCurrentSignal(instrument)
	if sig == nil {
		return 0
	}
	return baseSize * sig.RiskScalar
}

func runOTPStrategyConsumer() {
	brokers := os.Getenv("KAFKA_BROKERS")
	if brokers == "" {
		brokers = "localhost:9092"
	}

	consumer := NewRegimeConsumer(brokers, "coherence.signals.v1")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// In a real OTP service, this runs alongside the strategy engine:
	//   go consumer.Run(ctx)
	//   for {
	//       sig := consumer.GetCurrentSignal("EURUSD")
	//       if consumer.ShouldBlockNewPositions("EURUSD") {
	//           continue // wait for favorable regime
	//       }
	//       size := consumer.RiskScaledSize("EURUSD", basePosition)
	//       orderRouter.Submit(instrument, side, size)
	//   }

	if err := consumer.Run(ctx); err != nil {
		log.Fatal(err)
	}
}
