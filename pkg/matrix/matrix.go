package matrix

import "math/rand"

type Matrix struct {
	Rows, Cols int
	Data       [][]float64
}

func NewMatrix(rows, cols int) *Matrix {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
		for j := range data[i] {
			data[i][j] = 0
		}
	}
	return &Matrix{rows, cols, data}
}

func (m *Matrix) Randomize() {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] = 2*rand.Float64() - 1
		}
	}
}

func (m *Matrix) Add(n *Matrix) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] += n.Data[i][j]
		}
	}
}

func ApplyFunc(m *Matrix, f func(float64) float64) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] = f(m.Data[i][j])
		}
	}
}

func ApplyFuncToNew(m *Matrix, f func(float64) float64) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = f(m.Data[i][j])
		}
	}
	return result
}

func SliceToMatrix(s []float64) *Matrix {
	m := NewMatrix(len(s), 1)
	for i := range s {
		m.Data[i][0] = s[i]
	}
	return m
}

func Multiply(a, b *Matrix) *Matrix {
	if a.Cols != b.Rows {
		panic("columns of A must match rows of B for multiplication!")
	}

	result := NewMatrix(a.Rows, b.Cols)
	for i := 0; i < result.Rows; i++ {
		for j := 0; j < result.Cols; j++ {
			sum := 0.0
			for k := 0; k < a.Cols; k++ {
				sum += a.Data[i][k] * b.Data[k][j]
			}
			result.Data[i][j] = sum
		}
	}
	return result
}

func Subtract(a, b *Matrix) *Matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("both matrices must have the same dimensions for subtraction!")
	}

	result := NewMatrix(a.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			result.Data[i][j] = a.Data[i][j] - b.Data[i][j]
		}
	}
	return result
}

func (m *Matrix) HadamardMultiply(n *Matrix) {
	if m.Rows != n.Rows || m.Cols != n.Cols {
		panic("both matrices must have the same dimensions for Hadamard multiplication!")
	}

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] *= n.Data[i][j]
		}
	}
}

func (m *Matrix) ScalarMultiply(val float64) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] *= val
		}
	}
}

func Transpose(m *Matrix) *Matrix {
	result := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[j][i] = m.Data[i][j]
		}
	}
	return result
}
