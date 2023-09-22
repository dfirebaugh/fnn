package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"fnn/pkg/neuralnet"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"time"
)

func init() {
	gob.Register(&neuralnet.NeuralNetwork{})
}

func clearTerminal() {
	var cmd *exec.Cmd
	switch os := runtime.GOOS; os {
	case "linux", "darwin":
		cmd = exec.Command("clear")
	case "windows":
		cmd = exec.Command("cls")
	default:
		return
	}
	cmd.Stdout = os.Stdout
	cmd.Run()
}

func computeMSELoss(target, prediction []float64) float64 {
	sum := 0.0
	for i, t := range target {
		diff := t - prediction[i]
		sum += diff * diff
	}
	return sum / float64(len(target))
}

var (
	andTrainingData = []struct {
		input  []float64
		target []float64
	}{
		{input: []float64{0, 0}, target: []float64{0}},
		{input: []float64{0, 1}, target: []float64{0}},
		{input: []float64{1, 0}, target: []float64{0}},
		{input: []float64{1, 1}, target: []float64{1}},
	}

	orTrainingData = []struct {
		input  []float64
		target []float64
	}{
		{input: []float64{0, 0}, target: []float64{0}},
		{input: []float64{0, 1}, target: []float64{1}},
		{input: []float64{1, 0}, target: []float64{1}},
		{input: []float64{1, 1}, target: []float64{1}},
	}

	nandTrainingData = []struct {
		input  []float64
		target []float64
	}{
		{input: []float64{0, 0}, target: []float64{1}},
		{input: []float64{0, 1}, target: []float64{1}},
		{input: []float64{1, 0}, target: []float64{1}},
		{input: []float64{1, 1}, target: []float64{0}},
	}

	norTrainingData = []struct {
		input  []float64
		target []float64
	}{
		{input: []float64{0, 0}, target: []float64{1}},
		{input: []float64{0, 1}, target: []float64{0}},
		{input: []float64{1, 0}, target: []float64{0}},
		{input: []float64{1, 1}, target: []float64{0}},
	}

	xorTrainingData = []struct {
		input  []float64
		target []float64
	}{
		{input: []float64{0, 0}, target: []float64{0}},
		{input: []float64{0, 1}, target: []float64{1}},
		{input: []float64{1, 0}, target: []float64{1}},
		{input: []float64{1, 1}, target: []float64{0}},
	}
)

func train(nn *neuralnet.NeuralNetwork) {
	trainingData := xorTrainingData

	maxEpochs := 10000000000
	startTime := time.Now()
	const printInterval = 1000

	for epoch := 1; epoch <= maxEpochs; epoch++ {
		for _, data := range trainingData {
			nn.Train(data.input, data.target)
		}
		totalLoss := 0.0
		correctPredictions := true
		for _, data := range trainingData {
			output := nn.FeedForward(data.input)
			prediction := round(output.Data[0][0])
			totalLoss += prediction

			if prediction != data.target[0] {
				correctPredictions = false
				break
			}
		}

		avgLoss := totalLoss / float64(len(trainingData))

		if epoch%printInterval == 0 {
			endTime := time.Now()
			epochDuration := endTime.Sub(startTime).Seconds() / printInterval
			clearTerminal()
			fmt.Printf("Epoch %d to %d took average %.2f seconds per epoch - Avg Loss: %.4f\n", epoch-printInterval+1, epoch, epochDuration, avgLoss)

			startTime = time.Now()
		}

		if correctPredictions {
			break
		}
	}

	fmt.Println("Neural network trained successfully!")

	err := nn.SaveToFile("model.gob")
	if err != nil {
		fmt.Println("Failed to save the trained model:", err)
		return
	}
	fmt.Println("Trained model saved successfully.")
}
func round(f float64) float64 {
	if f < 0.5 {
		return 0
	}
	return 1
}
func testNN(nn *neuralnet.NeuralNetwork) {
	fmt.Println("Testing Neural Network:")

	args := flag.Args()
	if len(args) != 2 {
		fmt.Println("Please provide two float inputs for testing.")
		return
	}

	input1, err1 := strconv.ParseFloat(args[0], 64)
	input2, err2 := strconv.ParseFloat(args[1], 64)
	if err1 != nil || err2 != nil {
		fmt.Println("Failed to parse inputs. Please provide valid floats.")
		return
	}

	output := nn.FeedForward([]float64{input1, input2})
	fmt.Printf("Input: [%f, %f] - Actual: %d\n", input1, input2, int(round(output.Data[0][0])))
}

func main() {
	trainFlag := flag.Bool("train", false, "Set this flag to train the neural network.")
	testFlag := flag.Bool("test", false, "Set this flag to test the neural network on given inputs.")
	flag.Parse()

	if *trainFlag == *testFlag {
		fmt.Println("Please specify only one flag: -train or -test.")
		return
	}

	activation := &neuralnet.Sigmoid{}
	loss := &neuralnet.MSE{}

	config := &neuralnet.Config{
		InputNodes:   2,
		HiddenNodes:  5,
		OutputNodes:  1,
		LearningRate: 0.1,
		Activation:   activation,
		Loss:         loss,
	}

	nn, err := neuralnet.LoadFromFile("model.gob", config)
	if err != nil {
		fmt.Println("Error loading model:", err)
		return
	}

	if *trainFlag {
		train(nn)
	}

	if *testFlag {
		testNN(nn)
	}
}
