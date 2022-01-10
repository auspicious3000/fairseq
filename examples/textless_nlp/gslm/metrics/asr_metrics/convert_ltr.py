import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert Librispeech manifest format to asr metric format")
    parser.add_argument('--transcript', type=str,
                        help='Path to the transcript file.')
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    with open(args.transcript, 'r') as fi, open(args.output, 'w') as fo:
        for i, l in enumerate(fi):
            line = l.strip()
            trans = ''.join(line.split()).replace('|', ' ').upper().strip()
            new_line = f'{trans} (None-{i})\n'
            fo.write(new_line)
            
        
