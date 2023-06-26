import argparse
import logging
import json

from accelerate.logging import get_logger
import lang2vec.lang2vec as l2v

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Get lang2vec distances between languages")
    parser.add_argument(
        "--languages",
        type=str,
        help="List of languages to calculate distances for (two letter codes)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name",
        choices=["udpos", "PAN-X", "xnli", "tydiqa"],
        default="udpos"
    )
    parser.add_argument(
        "--l2v_code_map_path",
        type=str,
        help="Path to l2v code_map.json",
        default="src/helper/l2v/code_map.json"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path",
        default="src/helper/l2v"
    )
    args = parser.parse_args()
    return args

def main():
    """ Main function."""
    args = parse_args()
    level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )

    # For UDPOS, we need to convert full names to two letter codes
    lang_map = {"Afrikaans": "af",
                "Arabic": "ar",
                "Basque":"eu",
                "Bulgarian":"bg",
                "Dutch":"nl",
                "English":"en",
                "Estonian":"et",
                "Finnish":"fi",
                "French":"fr",
                "German":"de",
                "Greek":"el",
                "Hebrew": "he",
                "Hindi":"hi",
                "Hungarian":"hu",
                "Indonesian":"id",
                "Italian":"it",
                "Japanese":"ja",
                "Korean":"ko",
                "Chinese":"zh",
                "Marathi":"mr",
                "Persian":"fa",
                "Portuguese":"pt",
                "Russian":"ru",
                "Spanish":"es",
                "Tamil": "ta",
                "Telugu":"te",
                "Turkish":"tr",
                "Vietnamese":"vi",
                "Urdu":"ur"}

    if args.dataset_name == "udpos":
        languages = []
        for lang in lang_map.keys():
            languages.append(lang_map[lang])
    else:
        languages = args.languages.split(",")
    
    # Read l2v_code_map.json
    with open(args.l2v_code_map_path, "r") as fp:
        l2v_code_map = json.load(fp)

    # Write output to file
    fp = open(f"{args.output_path}/{args.dataset_name}_l2v_distances.txt", "w")

    # Make pairs of all languages and calculate syntactic distances
    for src in languages:
        for tgt in languages:
            if src not in l2v_code_map.keys() or tgt not in l2v_code_map.keys() or src == tgt:
                continue
            logger.info(f"Calculating distances between {src} and {tgt}")
            src_3_code = l2v_code_map[src]
            tgt_3_code = l2v_code_map[tgt]
            fp.write(f"{src}\t{tgt}\tsyntactic_distance\t{l2v.syntactic_distance(src_3_code, tgt_3_code)}\n")
            fp.write(f"{src}\t{tgt}\tgeographic_distance\t{l2v.geographic_distance([src_3_code, tgt_3_code])}\n")
            fp.write(f"{src}\t{tgt}\tphonological_distance\t{l2v.phonological_distance(src_3_code, tgt_3_code)}\n")
            fp.write(f"{src}\t{tgt}\tgenetic_distance\t{l2v.genetic_distance(src_3_code, tgt_3_code)}\n")
            fp.write(f"{src}\t{tgt}\tinventory_distance\t{l2v.inventory_distance(src_3_code, tgt_3_code)}\n")
            fp.write(f"{src}\t{tgt}\tfeatural_distance\t{l2v.featural_distance(src_3_code, tgt_3_code)}\n")
    fp.close()


if __name__ == "__main__":
    main()