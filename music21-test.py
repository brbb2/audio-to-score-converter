import music21


def main():
    s = music21.converter.parse("tinynotation: 3/4 c4 d8 f g16 a g f# a2 r4")
    print(s)


if __name__ == "__main__":
    main()
