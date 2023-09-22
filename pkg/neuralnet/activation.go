package neuralnet

import (
	"fnn/pkg/matrix"
	"math"
)

type Sigmoid struct{}

func (s *Sigmoid) Activate(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
func (s *Sigmoid) Derivative(x float64) float64 {
	return x * (1.0 - x)
}

type ReLU struct{}

func (r *ReLU) Activate(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func (r *ReLU) Derivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

type MSE struct{}

func (m *MSE) Compute(targets, outputs *matrix.Matrix) float64 {
	sum := 0.0
	for i := 0; i < targets.Rows; i++ {
		for j := 0; j < targets.Cols; j++ {
			diff := targets.Data[i][j] - outputs.Data[i][j]
			sum += diff * diff
		}
	}
	return sum / float64(targets.Rows)
}

func (m *MSE) Derivative(targets, outputs *matrix.Matrix) *matrix.Matrix {
	derivatives := matrix.NewMatrix(targets.Rows, targets.Cols)

	for i := 0; i < targets.Rows; i++ {
		for j := 0; j < targets.Cols; j++ {
			derivatives.Data[i][j] = outputs.Data[i][j] - targets.Data[i][j]
		}
	}

	return derivatives
}
