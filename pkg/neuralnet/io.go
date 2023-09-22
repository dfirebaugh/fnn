package neuralnet

import (
	"encoding/gob"
	"os"
)

func init() {
	gob.Register(Sigmoid{})
	gob.Register(MSE{})
	gob.Register(ReLU{})
}

func (nn *NeuralNetwork) SaveToFile(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(nn)
	if err != nil {
		return err
	}

	return nil
}

func LoadFromFile(filename string, c *Config) (*NeuralNetwork, error) {
	file, err := os.OpenFile(filename, os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil {
		return nil, err
	}

	if info.Size() == 0 {
		nn := New(c)
		return nn, nil
	}

	decoder := gob.NewDecoder(file)
	var nn NeuralNetwork
	err = decoder.Decode(&nn)
	if err != nil {
		return nil, err
	}

	return &nn, nil
}
