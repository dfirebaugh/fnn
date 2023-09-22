package neuralnet

import (
	"fnn/pkg/matrix"
)

type ActivationFunc interface {
	Activate(x float64) float64
	Derivative(x float64) float64
}

type LossFunc interface {
	Compute(targets, outputs *matrix.Matrix) float64
	Derivative(targets, outputs *matrix.Matrix) *matrix.Matrix
}
