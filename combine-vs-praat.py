# Combines VoiceSauce and Praat data for a given language
# Includes only B M C
# For eng, excludes stop words
# For eng, excludes stop words
# For eng, converts surrounding phones into six binary features
# Saves data as new csv

# Arguments:
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('stopwords', type=str,
                        help='path to the list of stopwords')
    parser.add_argument('lg', type=str, help='the three digit code for the language in question')
    return parser.parse_args()

def main():
	args = parse_args()

if __name__ == "__main__":
    main()