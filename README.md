# fnn
This is a new concept for me, but this repo is me playing around with some neural net stuff.

some concepts from https://nnfs.io/

## train
```bash
go run ./cmd/main/ -train
```

A model will be created as a `.gob` file.
> note: if you switch binary operations, you will probably want to delete any existing model before training

## test
```bash
go run ./cmd/main/ 1 1
```
(should print out what it thinks the xor of 1 1 is...)
I tried out a few other binary operations (switching between binary operations is currently hard coded).
