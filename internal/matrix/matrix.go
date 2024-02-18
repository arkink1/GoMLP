package matrix

import (
	"fmt"
	"math"
	"sync"
)

type Matrix []Vector

func (f Matrix) DeepCopy() Matrix {
	out := make(Matrix, len(f))
	var wg sync.WaitGroup
	for i := range f {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			out[i] = make(Vector, len(f[i]))
			copy(out[i], f[i])
		}(i)
	}
	wg.Wait()
	return out
}

func (f Matrix) Apply(fn func(float32) float32) {
	var wg sync.WaitGroup
	for i := range f {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := range f[i] {
				f[i][j] = fn(f[i][j])
			}
		}(i)
	}
	wg.Wait()
}

func (f Matrix) ForEach(fn func(float32)) {
	var wg sync.WaitGroup
	for i := range f {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := range f[i] {
				fn(f[i][j])
			}
		}(i)
	}
	wg.Wait()
}

func (f Matrix) ForEachPairwise(o Matrix, fn func(float32, float32)) {
	var wg sync.WaitGroup
	for i := range f {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := range f[i] {
				fn(f[i][j], o[i][j])
			}
		}(i)
	}
	wg.Wait()
}

func (f Matrix) Pairwise(o Matrix, fn func(float32, float32) float32) Matrix {
	var wg sync.WaitGroup
	out := f.DeepCopy()
	for i := range f {
		wg.Add(1)
		go func(i int) {
			for j := range f[i] {
				out[i][j] = fn(f[i][j], o[i][j])
			}
		}(i)
	}
	wg.Wait()
	return out
}

type Vector []float32

func (v Vector) DeepCopy() Vector {
	out := make(Vector, len(v))
	copy(out, v)
	return out
}

func (v Vector) Apply(fn func(float32) float32) Vector {
	out := v.DeepCopy()
	for i := range out {
		out[i] = fn(out[i])
	}
	return out
}

func (v Vector) Scalar(s float32) Vector {
	return v.Apply(func(e float32) float32 {
		return e * s
	})
}

func (v Vector) Subtract(o Vector) Vector {
	out := v.DeepCopy()
	for i := range v {
		out[i] = v[i] - o[i]
	}
	return out
}

func (v Vector) ElementwiseProduct(o Vector) Vector {
	out := v.DeepCopy()
	for i := range v {
		out[i] = v[i] * o[i]
	}
	return out
}

func DotProduct(a, b Vector) float32 {
	if len(a) != len(b) {
		panic(fmt.Errorf(
			"cannot dot product arrays of unequal length: %d, %d",
			len(a),
			len(b),
		))
	}
	var res float32
	for i := range a {
		res += a[i] * b[i]
	}
	return res
}

func (v Vector) MaxVal() int {
	var max float32 = -math.MaxFloat32
	var imax int
	for i, val := range v {
		if val > max {
			max = val
			imax = i
		}
	}
	return imax
}
