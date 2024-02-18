package neural

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"matrix"
)

type MLP struct {
	Layers       []*Layer
	LearningRate float32
	Status       func(step Step)
}

type Step struct {
	Epoch int
	Loss  float32
}

type Layer struct {
	Name                string
	Neurons             int
	ActivationFunc      func(float32) float32
	ActivationFuncDeriv func(float32) float32
	MlpPtr              *MLP
	Prev                *Layer
	Next                *Layer
	initialized         bool
	weights             matrix.Matrix
	biases              matrix.Vector
	lastZ               matrix.Vector
	lastActivations     matrix.Vector
	lastError           matrix.Vector
	lastLoss            matrix.Matrix
}

func (n *MLP) InitializeNN() {
	var Prev *Layer
	for i, layer := range n.Layers {
		var Next *Layer
		if i < len(n.Layers)-1 {
			Next = n.Layers[i+1]
		}
		layer.Initialize(n, Prev, Next)
		Prev = layer
	}
}

func (n *MLP) Train(epochs int, inputs, labels matrix.Matrix) (float32, error) {
	if err := n.Check(inputs, labels); err != nil {
		return 0, err
	}

	n.InitializeNN()

	var loss float32
	for e := 0; e < epochs; e++ {
		predictions := make(matrix.Matrix, len(inputs))

		for i, input := range inputs {
			activations := input
			for _, layer := range n.Layers {
				activations = layer.ForwardProp(activations)
			}
			predictions[i] = activations

			for step := range n.Layers {
				l := len(n.Layers) - (step + 1)
				layer := n.Layers[l]

				if l == 0 {
					continue
				}

				layer.BackProp(labels[i])
			}
		}

		loss = Loss(predictions, labels)
		if n.Status != nil {
			n.Status(Step{
				Epoch: e,
				Loss:  loss,
			})
		}

	}

	return loss, nil
}

func (n *MLP) Predict(inputs matrix.Matrix) matrix.Matrix {
	preds := make(matrix.Matrix, len(inputs))
	for i, input := range inputs {
		activations := input
		for _, layer := range n.Layers {
			activations = layer.ForwardProp(activations)
		}
		preds[i] = activations
	}
	return preds
}

func (n *MLP) Check(inputs matrix.Matrix, outputs matrix.Matrix) error {
	if len(n.Layers) == 0 {
		return errors.New("network must have at least one layer")
	}

	if len(inputs) != len(outputs) {
		return fmt.Errorf("inputs count %d mismatched with outputs count %d", len(inputs), len(outputs))
	}
	return nil
}

func (l *Layer) Initialize(MlpPtr *MLP, Prev *Layer, Next *Layer) {
	if l.initialized || Prev == nil {
		return
	}
	l.MlpPtr = MlpPtr
	l.Prev = Prev
	l.Next = Next
	if l.ActivationFunc == nil {
		l.ActivationFunc = Sigmoid
	}
	if l.ActivationFuncDeriv == nil {
		l.ActivationFuncDeriv = SigmoidDeriv
	}
	l.weights = make(matrix.Matrix, l.Neurons)
	for i := range l.weights {
		l.weights[i] = make(matrix.Vector, l.Prev.Neurons)
		for j := range l.weights[i] {
			weight := rand.NormFloat64() * math.Pow(float64(l.Prev.Neurons), -0.5)
			l.weights[i][j] = float32(weight)
		}
	}
	l.biases = make(matrix.Vector, l.Neurons)
	for i := range l.biases {
		l.biases[i] = rand.Float32()
	}
	l.lastError = make(matrix.Vector, l.Neurons)
	l.lastLoss = make(matrix.Matrix, l.Neurons)
	for i := range l.lastLoss {
		l.lastLoss[i] = make(matrix.Vector, l.Prev.Neurons)
	}

	l.initialized = true
}

func (l *Layer) ForwardProp(input matrix.Vector) matrix.Vector {
	if l.Prev == nil {
		l.lastActivations = input
		return input
	}

	Z := make(matrix.Vector, l.Neurons)
	activations := make(matrix.Vector, l.Neurons)
	for i := range activations {
		nodeWeights := l.weights[i]
		nodeBias := l.biases[i]
		Z[i] = matrix.DotProduct(input, nodeWeights) + nodeBias
		activations[i] = l.ActivationFunc(Z[i])
	}
	l.lastZ = Z
	l.lastActivations = activations
	return activations
}

func (l *Layer) BackProp(label matrix.Vector) {
	if l.Next == nil {
		l.lastError = l.lastActivations.Subtract(label)
	} else {
		l.lastError = make(matrix.Vector, len(l.lastError))
		for j := range l.weights {
			for jn := range l.Next.lastLoss {
				l.lastError[j] += l.Next.lastLoss[jn][j]
			}
		}
	}
	dLdA := l.lastError.Scalar(2)

	dAdZ := l.lastZ.Apply(l.ActivationFuncDeriv)

	for j := range l.weights {
		l.lastLoss[j] = l.weights[j].Scalar(l.lastError[j])
	}

	for j := range l.weights {
		for k := range l.weights[j] {
			dZdW := l.Prev.lastActivations[k]
			dLdW := dLdA[j] * dAdZ[j] * dZdW
			l.weights[j][k] -= dLdW * l.MlpPtr.LearningRate
		}
	}

	biasUpdate := dLdA.ElementwiseProduct(dAdZ)
	l.biases = l.biases.Subtract(biasUpdate.Scalar(l.MlpPtr.LearningRate))
}

func (m *MLP) PredictionTestOneHot(inputs, labels matrix.Matrix) float32 {
	results := make([]bool, len(labels))
	for i, predictionOH := range m.Predict(inputs) {
		prediction := predictionOH.MaxVal()
		label := labels[i].MaxVal()

		results[i] = prediction == label
	}

	var score float32
	for _, r := range results {
		if r {
			score += 1.0
		}
	}
	score = score / float32(len(results))
	return score
}
