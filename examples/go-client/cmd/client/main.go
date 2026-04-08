package main

import (
	"context"
	"os"

	goclient "github.com/neuron7xLab/GeoSync/examples/go-client"
)

func main() {
	ctx, cancel := goclient.ContextWithSignals(context.Background())
	defer cancel()

	grpcTarget := os.Getenv("COHERENCE_GRPC_TARGET")
	if grpcTarget == "" {
		grpcTarget = "localhost:50051"
	}
	healthURL := os.Getenv("COHERENCE_HTTP_BASE")
	if healthURL == "" {
		healthURL = "http://localhost:8080"
	}

	_ = goclient.Run(ctx, grpcTarget, healthURL, goclient.NewJSONLogger())
}
