package neuralnet

import (
	"bytes"
	"encoding/gob"
	"errors"
)

func (nn *NeuralNetwork) GobEncode() ([]byte, error) {
	w := new(bytes.Buffer)
	encoder := gob.NewEncoder(w)

	err := encoder.Encode(nn.InputNodes)
	if err != nil {
		return nil, err
	}

	err = encoder.Encode(nn.HiddenNodes)
	if err != nil {
		return nil, err
	}

	err = encoder.Encode(nn.OutputNodes)
	if err != nil {
		return nil, err
	}

	err = encoder.Encode(nn.WeightsIH)
	if err != nil {
		return nil, err
	}

	err = encoder.Encode(nn.WeightsHO)
	if err != nil {
		return nil, err
	}

	err = encoder.Encode(nn.BiasH)
	if err != nil {
		return nil, err
	}

	err = encoder.Encode(nn.BiasO)
	if err != nil {
		return nil, err
	}

	err = encoder.Encode(nn.LearningRate)
	if err != nil {
		return nil, err
	}

	switch v := nn.Activation.(type) {
	case *Sigmoid:
		err = encoder.Encode("Sigmoid")
		if err != nil {
			return nil, err
		}
		err = encoder.Encode(v)
		if err != nil {
			return nil, err
		}
	case *ReLU:
	default:
		return nil, errors.New("unsupported activation type")
	}

	return w.Bytes(), nil
}

func (nn *NeuralNetwork) GobDecode(buf []byte) error {
	r := bytes.NewBuffer(buf)
	decoder := gob.NewDecoder(r)

	err := decoder.Decode(&nn.InputNodes)
	if err != nil {
		return err
	}

	err = decoder.Decode(&nn.HiddenNodes)
	if err != nil {
		return err
	}
	err = decoder.Decode(&nn.OutputNodes)
	if err != nil {
		return err
	}

	err = decoder.Decode(&nn.WeightsIH)
	if err != nil {
		return err
	}

	err = decoder.Decode(&nn.WeightsHO)
	if err != nil {
		return err
	}

	err = decoder.Decode(&nn.BiasH)
	if err != nil {
		return err
	}

	err = decoder.Decode(&nn.BiasO)
	if err != nil {
		return err
	}

	err = decoder.Decode(&nn.LearningRate)
	if err != nil {
		return err
	}

	var activationType string
	err = decoder.Decode(&activationType)
	if err != nil {
		return err
	}

	switch activationType {
	case "Sigmoid":
		activation := &Sigmoid{}
		err = decoder.Decode(activation)
		if err != nil {
			return err
		}
		nn.Activation = activation
	case "ReLU":
		// not implemented
	default:
		return errors.New("unsupported activation type")
	}

	return nil
}
