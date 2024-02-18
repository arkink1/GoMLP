package mnist

import (
	"matrix"
	"path/filepath"
)

type MNIST struct {
	TrainInputs matrix.Matrix
	TrainLabels matrix.Matrix
	TestInputs  matrix.Matrix
	TestLabels  matrix.Matrix
}

func ReadR(rootDir string) (*MNIST, error) {
	trainRawImages, err := ReadFile(
		filepath.Join(rootDir, "train-images-idx3-ubyte.gz"))
	if err != nil {
		return nil, err
	}
	trainRawLabels, err := ReadFile(
		filepath.Join(rootDir, "train-labels-idx1-ubyte.gz"))
	if err != nil {
		return nil, err
	}

	testRawImages, err := ReadFile(
		filepath.Join(rootDir, "t10k-images-idx3-ubyte.gz"))
	if err != nil {
		return nil, err
	}
	testRawLabels, err := ReadFile(
		filepath.Join(rootDir, "t10k-labels-idx1-ubyte.gz"))
	if err != nil {
		return nil, err
	}

	out := &MNIST{
		TrainInputs: make(matrix.Matrix, trainRawImages.Dimensions[0]),
		TrainLabels: make(matrix.Matrix, trainRawLabels.Dimensions[0]),
		TestInputs:  make(matrix.Matrix, testRawImages.Dimensions[0]),
		TestLabels:  make(matrix.Matrix, testRawLabels.Dimensions[0]),
	}

	for i := range out.TrainInputs {
		out.TrainInputs[i] = make([]float32, 28*28)
		for j := range out.TrainInputs[i] {
			out.TrainInputs[i][j] = float32(
				trainRawImages.Data[i*28*28+j])/255.0*0.99 + 0.01
		}
	}
	for i := range out.TestInputs {
		out.TestInputs[i] = make([]float32, 28*28)
		for j := range out.TestInputs[i] {
			out.TestInputs[i][j] = float32(
				testRawImages.Data[i*28*28+j])/255.0*0.99 + 0.01
		}
	}

	for i := range out.TrainLabels {
		out.TrainLabels[i] = make([]float32, 10)
		for j := range out.TrainLabels[i] {
			out.TrainLabels[i][j] = 0.01
		}
		out.TrainLabels[i][trainRawLabels.Data[i]] = 0.99
	}
	for i := range out.TestLabels {
		out.TestLabels[i] = make([]float32, 10)
		for j := range out.TestLabels[i] {
			out.TestLabels[i][j] = 0.01
		}
		out.TestLabels[i][testRawLabels.Data[i]] = 0.99
	}

	return out, nil
}
