# DCGAN Example

![Example(10 epochs)](https://raw.githubusercontent.com/ryokobachan/DCGAN/master/result.png)

## License

This software is released under the MIT License.

## Requirements

```
$ pip install -r requirements.txt

```

## Usage

Use mnist RGB dataset(Default)
```
$ python main.py
```

Use mnist Grayscale dataset
```
$ python main.py -t mnistgray
```

Use custom images dataset from './DIR_PATH'
```
$ python main.py -t DIR_PATH
```
