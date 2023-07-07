# UzSyllable

https://pypi.org/project/UzSyllable <br>
https://github.com/UlugbekSalaev/UzSyllable

UzSyllable tool is focused to make division of syllables and end of line breaks of Uzbek language. The tool includes Syllabification, End-of-Line, Count of Syllables methods.
It is created as a python library and uploaded to [PyPI](https://pypi.org/). It is simply easy to use in your python project or other programming language projects via the API. 

## About project
The UzSyllable project is a text processing tool that includes three main methods: syllabification, end-of-line, and count of syllables. Syllabification involves dividing words in Uzbek text into their constituent syllables, which can be useful for pronunciation, spelling, and linguistic analysis. End-of-line justification involves determining the appropriate places to break lines in Uzbek text, which can improve the readability and aesthetics of written materials. Count of syllables involves counting the number of syllables in a given word or sentence, which can be useful for metrics such as rhyme and meter in poetry or for determining the complexity of a text. The UzSyllable project uses machine learning algorithms and linguistic rules to perform these methods accurately and efficiently on Uzbek text.
## Quick links

- [Github](https://github.com/UlugbekSalaev/UzSyllable)
- [PyPI](https://pypi.org/project/UzSyllable/)
- [Web-UI](https://nlp.urdu.uz/?menu=uzsyllable)

## Demo

You can use [web interface](http://nlp.urdu.uz/?menu=uzsyllable).

## Features

- Syllabification
- Hyphenation
- Count of Syllables

## Usage

Two options to run UzSyllable:

- pip
- Web interface

### pip installation

To install UzSyllable, simply run:

```code
pip install UzSyllable
```

After installation, use in python like following:

Syllabification
```code
from UzSyllable import syllables
print(syllables('maktabimda'))
# Output : ['mak-ta-bim-da']

print(syllables('мактабимда'))
# Output : ['мак-та-бим-да']
```

Hyphenation
```code
from UzSyllable import hyphenation
# call end-of-line method
print(hyphenation('maktabimda'))
# Output : ['mak-tabimda', 'makta-bimda', 'maktabim-da']
```

Count of Syllables
```code
from UzSyllable import count
# call count of syllables method
print(count('maktabimda'))
# Output : 4
```


## Documentation

See [here](https://github.com/UlugbekSalaev/UzSyllable).

## Citation

```tex
@misc{UzSyllable,
  title={UzSyllable}: Syllabification Tool for Uzbek},
  url={https://github.com/UlugbekSalaev/UzSyllable},
  note={Software available from https://github.com/UlugbekSalaev/UzSyllable},
  author={
    Ulugbek Salaev},
  year={2023},
}
```

## Contact

For help and feedback, please feel free to contact [the author](https://github.com/UlugbekSalaev).
