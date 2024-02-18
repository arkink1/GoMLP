// MNIST parser

package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

type IDX struct {
	Dimensions []uint32
	Data       []uint8
}

func ReadFile(filename string) (*IDX, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var rdr io.Reader = f
	if filepath.Ext(filename) == ".gz" {
		gzrdr, err := gzip.NewReader(rdr)
		if err != nil {
			return nil, err
		}
		rdr = gzrdr
	}

	return Read(rdr)
}

func Read(rdr io.Reader) (*IDX, error) {
	magic := make([]byte, 4)
	n, err := rdr.Read(magic)
	if err != nil {
		return nil, err
	}
	if n != len(magic) {
		return nil, fmt.Errorf("read %d bytes for magic number, expected 4", n)
	}

	if magic[0] != 0 || magic[1] != 0 {
		return nil, fmt.Errorf(
			"first two bytes of magic number should be zero, got: %q", magic)
	}

	if magic[2] != 0x08 {
		return nil, fmt.Errorf(
			"only uint8 data type supported, got: %x", magic[2])
	}

	ndim := magic[3]

	o := &IDX{
		Dimensions: make([]uint32, ndim),
	}

	if err := binary.Read(rdr, binary.BigEndian, &o.Dimensions); err != nil {
		return nil, err
	}

	var totalLen uint32 = 1
	for _, d := range o.Dimensions {
		totalLen = totalLen * d
	}

	o.Data = make([]uint8, int(totalLen))
	if err := binary.Read(rdr, binary.BigEndian, &o.Data); err != nil {
		return nil, err
	}

	return o, nil
}
