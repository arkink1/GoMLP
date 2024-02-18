package neural

import (
	"math"
	"matrix"
)

func Sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(-float64(x))))
}

func SigmoidDeriv(x float32) float32 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

func Loss(pred, labels matrix.Matrix) float32 {
	var squaredError, count float32
	pred.ForEachPairwise(labels, func(o, l float32) {
		count += 1.0
		squaredError += (o - l) * (o - l)
	})
	return squaredError / count
}

func Multiply(x, y float32) float32 {
	return x * y
}
