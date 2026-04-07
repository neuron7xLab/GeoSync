// CoherenceBridge gRPC client example for OTP integration.
//
// OTP service integration points (from ettec/open-trading-platform):
//   go/execution-venues/order-router  — SmartRouter consumes regime for venue selection
//   go/vwap-strategy                  — VWAP adjusts slice sizes via risk_scalar
//   go/order-monitor                  — exposes regime state to React UI
//   go/market-data-service            — can forward regime alongside quotes
//
// Shared domain models live in ettec/otp-common (Go module).
// Kafka topics in OTP: orders, trades, market-data.
// Our topic: coherence.signals.v1 (sits alongside these).
package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	cbv1 "github.com/neuron7xLab/coherence-bridge/proto/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	addr := "localhost:50051"
	if v := os.Getenv("BRIDGE_ADDR"); v != "" {
		addr = v
	}

	conn, err := grpc.NewClient(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		log.Fatalf("connect: %v", err)
	}
	defer conn.Close()

	client := cbv1.NewCoherenceBridgeClient(conn)

	// Health check
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	health, err := client.Health(ctx, &cbv1.Empty{})
	if err != nil {
		log.Fatalf("health: %v", err)
	}
	fmt.Printf("Server healthy=%v uptime=%ds signals=%d\n",
		health.Healthy, health.UptimeS, health.SignalsEmitted)

	// Stream signals with graceful shutdown
	streamCtx, streamCancel := context.WithCancel(context.Background())
	defer streamCancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Println("\nShutting down...")
		streamCancel()
	}()

	stream, err := client.StreamSignals(streamCtx, &cbv1.SignalRequest{
		Instruments:   []string{"EURUSD", "GBPUSD"},
		MinIntervalMs: 1000,
	})
	if err != nil {
		log.Fatalf("stream: %v", err)
	}

	for {
		sig, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			if streamCtx.Err() != nil {
				break // graceful shutdown
			}
			log.Fatalf("recv: %v", err)
		}
		fmt.Printf("[%s] γ=%.4f R=%.4f κ=%.4f λ=%.4f regime=%s conf=%.2f signal=%.4f risk=%.4f\n",
			sig.Instrument,
			sig.Gamma,
			sig.OrderParameterR,
			sig.RicciCurvature,
			sig.LyapunovMax,
			sig.Regime.String(),
			sig.RegimeConfidence,
			sig.SignalStrength,
			sig.RiskScalar,
		)
	}
}
