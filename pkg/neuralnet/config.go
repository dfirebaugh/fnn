package neuralnet

type Config struct {
	InputNodes   int
	HiddenNodes  int
	OutputNodes  int
	LearningRate float64

	Activation ActivationFunc
	Loss       LossFunc
}
