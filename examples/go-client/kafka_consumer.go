package main

// Kafka consumer example for OTP integration.
//
// This shows how Askar's OTP strategy framework can consume
// CoherenceBridge signals from Kafka alongside order flow events.
//
// Topic: coherence.signals.v1
// Key: instrument (UTF-8 string)
// Value: Protobuf RegimeSignal (same schema as gRPC)
//
// Build: go build -o kafka-consumer kafka_consumer.go
// Run:   KAFKA_BROKERS=localhost:9092 ./kafka-consumer
//
// Integration with OTP (ettec/open-trading-platform):
//   OTP Kafka topics: orders, trades, market-data
//   Our topic: coherence.signals.v1 (sits alongside these)
//
//   Target OTP services that consume this:
//     go/execution-venues/order-router — SmartRouter: block routing in CRITICAL regime
//     go/vwap-strategy                 — scale slice sizes by risk_scalar
//     go/order-monitor                 — expose regime to React UI via gRPC-web
//
//   Shared domain models: ettec/otp-common (Go module)
//
//   Example handler in OTP strategy:
//   func (s *Strategy) OnRegimeSignal(sig *cbv1.RegimeSignal) {
//       if sig.Regime == cbv1.REGIME_CRITICAL && sig.RiskScalar < 0.3 {
//           s.CancelAllOrders(sig.Instrument)
//       }
//       s.UpdatePositionSize(sig.Instrument, sig.RiskScalar)
//   }
//
// Dependencies:
//   go get github.com/IBM/sarama
//   go get google.golang.org/protobuf

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/IBM/sarama"
	cbv1 "github.com/neuron7xLab/coherence-bridge/proto/v1"
	"google.golang.org/protobuf/proto"
)

func runKafkaConsumer() {
	brokers := os.Getenv("KAFKA_BROKERS")
	if brokers == "" {
		brokers = "localhost:9092"
	}
	topic := "coherence.signals.v1"

	config := sarama.NewConfig()
	config.Consumer.Return.Errors = true

	consumer, err := sarama.NewConsumer([]string{brokers}, config)
	if err != nil {
		log.Fatalf("kafka connect: %v", err)
	}
	defer consumer.Close()

	partConsumer, err := consumer.ConsumePartition(topic, 0, sarama.OffsetNewest)
	if err != nil {
		log.Fatalf("kafka partition: %v", err)
	}
	defer partConsumer.Close()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	fmt.Printf("Consuming from %s on %s...\n", topic, brokers)

	for {
		select {
		case msg := <-partConsumer.Messages():
			sig := &cbv1.RegimeSignal{}
			if err := proto.Unmarshal(msg.Value, sig); err != nil {
				log.Printf("unmarshal error: %v", err)
				continue
			}
			fmt.Printf("[%s] γ=%.4f R=%.4f κ=%.4f λ=%.4f regime=%s risk=%.4f\n",
				sig.Instrument,
				sig.Gamma,
				sig.OrderParameterR,
				sig.RicciCurvature,
				sig.LyapunovMax,
				sig.Regime.String(),
				sig.RiskScalar,
			)

		case err := <-partConsumer.Errors():
			log.Printf("kafka error: %v", err)

		case <-sigCh:
			fmt.Println("\nShutting down...")
			return
		}
	}
}
