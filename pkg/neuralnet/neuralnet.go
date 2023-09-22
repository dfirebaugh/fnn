package neuralnet

import (
	"fnn/pkg/matrix"
	"math"
	"math/rand"
	"time"
)

type NeuralNetwork struct {
	InputNodes, HiddenNodes, OutputNodes int
	WeightsIH, WeightsHO                 *matrix.Matrix
	BiasH, BiasO                         *matrix.Matrix
	LearningRate                         float64
	Activation                           ActivationFunc
}

func New(config *Config) *NeuralNetwork {
	if config.InputNodes <= 0 || config.HiddenNodes <= 0 || config.OutputNodes <= 0 {
		panic("invalid configuration: Number of nodes in a layer must be positive!")
	}

	if config.Activation == nil || config.Loss == nil {
		panic("activation function and Loss function must be provided!")
	}

	rand.New(rand.NewSource(time.Now().UnixNano()))

	nn := &NeuralNetwork{
		InputNodes:   config.InputNodes,
		HiddenNodes:  config.HiddenNodes,
		OutputNodes:  config.OutputNodes,
		LearningRate: config.LearningRate,
		Activation:   config.Activation,
	}

	nn.WeightsIH = matrix.NewMatrix(config.HiddenNodes, config.InputNodes)
	nn.BiasH = matrix.NewMatrix(config.HiddenNodes, 1)
	initializeWeightsAndBiases(nn.WeightsIH, nn.BiasH, config.InputNodes)

	nn.WeightsHO = matrix.NewMatrix(config.OutputNodes, config.HiddenNodes)
	nn.BiasO = matrix.NewMatrix(config.OutputNodes, 1)
	initializeWeightsAndBiases(nn.WeightsHO, nn.BiasO, config.HiddenNodes)

	return nn
}

func initializeWeightsAndBiases(weights, biases *matrix.Matrix, prevLayerSize int) {
	factor := math.Sqrt(2.0 / float64(prevLayerSize))
	for i := 0; i < weights.Rows; i++ {
		for j := 0; j < weights.Cols; j++ {
			weights.Data[i][j] = rand.NormFloat64() * factor
		}
		biases.Data[i][0] = rand.NormFloat64() * factor
	}
}

func (nn *NeuralNetwork) FeedForward(inputArray []float64) *matrix.Matrix {
	inputs := matrix.NewMatrix(len(inputArray), 1)
	for i := range inputArray {
		inputs.Data[i][0] = inputArray[i]
	}

	hidden := matrix.Multiply(nn.WeightsIH, inputs)
	hidden.Add(nn.BiasH)
	for i := 0; i < hidden.Rows; i++ {
		for j := 0; j < hidden.Cols; j++ {
			hidden.Data[i][j] = nn.Activation.Activate(hidden.Data[i][j])
		}
	}

	output := matrix.Multiply(nn.WeightsHO, hidden)
	output.Add(nn.BiasO)
	for i := 0; i < output.Rows; i++ {
		for j := 0; j < output.Cols; j++ {
			output.Data[i][j] = nn.Activation.Activate(output.Data[i][j])
		}
	}

	return output
}

func (nn *NeuralNetwork) Train(inputArray []float64, targetArray []float64) {
	inputs := matrix.SliceToMatrix(inputArray)
	hidden := matrix.Multiply(nn.WeightsIH, inputs)
	hidden.Add(nn.BiasH)
	matrix.ApplyFunc(hidden, nn.Activation.Activate)
	outputs := matrix.Multiply(nn.WeightsHO, hidden)
	outputs.Add(nn.BiasO)
	matrix.ApplyFunc(outputs, nn.Activation.Activate)

	targets := matrix.SliceToMatrix(targetArray)

	outputErrors := matrix.Subtract(targets, outputs)

	gradients := matrix.ApplyFuncToNew(outputs, nn.Activation.Derivative)
	gradients.HadamardMultiply(outputErrors)
	gradients.ScalarMultiply(nn.LearningRate)

	hiddenT := matrix.Transpose(hidden)
	weightHODeltas := matrix.Multiply(gradients, hiddenT)

	nn.WeightsHO.Add(weightHODeltas)
	nn.BiasO.Add(gradients)

	weightsHOT := matrix.Transpose(nn.WeightsHO)
	hiddenErrors := matrix.Multiply(weightsHOT, outputErrors)

	hiddenGradients := matrix.ApplyFuncToNew(hidden, nn.Activation.Derivative)
	hiddenGradients.HadamardMultiply(hiddenErrors)
	hiddenGradients.ScalarMultiply(nn.LearningRate)

	inputsT := matrix.Transpose(inputs)
	weightIHDeltas := matrix.Multiply(hiddenGradients, inputsT)

	nn.WeightsIH.Add(weightIHDeltas)
	nn.BiasH.Add(hiddenGradients)
}
