import os

PATH_TIMIDITY = '/usr/local/bin'
PATH_MID = 'midi_files'
PATH_WAV = 'wav_files'


def run_timidity(filename):
    filename, _ = os.path.splitext(filename)
    command = '{}/timidity {}/{}.mid -Ow --preserve-silence -o {}/{}.wav'\
        .format(PATH_TIMIDITY, PATH_MID, filename, PATH_WAV, filename)
    os.system(command)


def make_all_wavs(skipping_rest_files=False):
    if skipping_rest_files:
        for filename in os.listdir(PATH_MID):
            if 'rest' in filename:
                continue
            else:
                run_timidity(filename)
    else:
        for filename in os.listdir(PATH_MID):
            run_timidity(filename)


def main():
    make_all_wavs()


if __name__ == '__main__':
    main()
