
# GoMLP: Multilayer Perceptron in Go with Goroutines

GoMLP is a simple implementation of a multilayer perceptron (MLP) neural network written from scratch in Go. This implementation utilizes Goroutines for concurrent computation, enabling faster training of the network.

The network has flexible architecture, allowing customization of network parameters such as the number of layers, neurons per layer, activation and loss functions, and learning rate.

This network achieved a 96.8% accuracy on the MNIST dataset. The network can be trained on the MNIST dataset using `main.go`.

## Install and Run

First clone the repository: `git clone https://github.com/arkink1/GoMLP.git`

Navigate to the `/cmd` folder.

Modify the `main.go` file to change the parameters of the MLP.

Run the file: `go run main.go`
