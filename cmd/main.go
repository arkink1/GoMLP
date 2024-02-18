package main

import (
	"fmt"
	"mnist"
	"neural"
)

func main() {
	fmt.Println("Go Neural Network")
	fmt.Println()

	mn, err := mnist.ReadR("../data")
	if err != nil {
		panic("invalid path to mnist data provided")
	}
	fmt.Printf("Train Inputs: %d x %d\n", len(mn.TrainInputs), len(mn.TrainInputs[0]))
	fmt.Printf("Train Labels: %d x %d\n", len(mn.TrainLabels), len(mn.TrainLabels[0]))
	fmt.Printf("Test Inputs: %d x %d\n", len(mn.TestInputs), len(mn.TestInputs[0]))
	fmt.Printf("Test Labels: %d x %d\n", len(mn.TestLabels), len(mn.TestLabels[0]))
	fmt.Println()

	const epochs = 5
	m := neural.MLP{
		LearningRate: 0.1,
		Layers: []*neural.Layer{
			{Name: "input", Neurons: 28 * 28},
			{Name: "hidden1", Neurons: 100},
			{Name: "output", Neurons: 10},
		},
		Status: func(s neural.Step) {
			fmt.Println("epoch:", s.Epoch+1, "/", epochs, "| loss:", s.Loss)
		},
	}

	fmt.Println("Starting training")
	loss, err := m.Train(epochs, mn.TrainInputs, mn.TrainLabels)
	if err != nil {
		panic("error while training")
	}
	fmt.Println("Training loss:", loss)
	fmt.Println()

	fmt.Println("Starting testing")
	accuracy := m.PredictionTestOneHot(mn.TestInputs, mn.TestLabels)
	fmt.Printf("Accuracy: %.2f%% \n", accuracy*100)
}
