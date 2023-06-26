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
    parser = argparse.ArgumentParser(description="Get languages with minimum distances for each candidate language")
    parser.add_argument(
        "--l2v_distances_file",
        type=str,
        help="Calculated l2v pairwise distances file",
        default="src/helper/l2v/udpos_l2v_distances.txt"
    )
    parser.add_argument(
        "--candidate_languages",
        type=str,
        help="Comma separated list of candidate languages for which we want to find minimum distances",
        default="te,et,el,fi,hu,mr,kk,hi,tr,eu,id"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path where we want to store the results",
        default="src/helper/l2v"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name",
        default="udpos"
    )
    args = parser.parse_args()
    return args


def get_min_dist(distances: dict[str, float]) -> dict[str, tuple[str, float]]:
    """ 
    Get minimum distance for each source language.

    Args:
        distances (dict[str, float]): Dictionary of distances between languages.
    
    Returns:
        dict[str, tuple[str, float]]: Dictionary of minimum distances for each source language.
    """
    distances_min = {}
    for source_language in distances.keys():
        min_value = 2 # anything greater than 1
        for target_language in distances[source_language].keys():
            if distances[source_language][target_language] < min_value and distances[source_language][target_language] > 0.05 and target_language!="en":
                min_value = distances[source_language][target_language]
                distances_min[source_language] = (target_language, min_value)
    return distances_min


def main():
    """ Main function."""
    args = parse_args()
    level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )
    
    # Read l2v_distances_files
    with open(args.l2v_distances_file, "r") as fp:
        l2v_distances = fp.readlines()
    
    syntactic_distances = {}
    phonological_distances = {}
    genetic_distances = {}
    geographical_distances = {}
    inventory_distances = {}
    
    candidate_languages = args.candidate_languages.split(",")
    for line in l2v_distances:
        source_language = line.split("\t")[0]
        target_language = line.split("\t")[1]
        if source_language in candidate_languages:
            if source_language not in syntactic_distances.keys():
                syntactic_distances[source_language] = {}
            if source_language not in phonological_distances.keys():
                phonological_distances[source_language] = {}
            if source_language not in genetic_distances.keys():
                genetic_distances[source_language] = {}
            if source_language not in geographical_distances.keys():
                geographical_distances[source_language] = {}
            if source_language not in inventory_distances.keys():
                inventory_distances[source_language] = {}
            if line.split("\t")[2] == "syntactic_distance":
                syntactic_distances[source_language][target_language] = float(line.split("\t")[3])
            elif line.split("\t")[2] == "phonological_distance":
                phonological_distances[source_language][target_language] = float(line.split("\t")[3])
            elif line.split("\t")[2] == "genetic_distance":
                genetic_distances[source_language][target_language] = float(line.split("\t")[3])
            elif line.split("\t")[2] == "geographic_distance":
                geographical_distances[source_language][target_language] = float(line.split("\t")[3])
            elif line.split("\t")[2] == "inventory_distance":
                inventory_distances[source_language][target_language] = float(line.split("\t")[3])
            
    # Get top-1 minimum distances and language with min distance too
    syntactic_distances_min = get_min_dist(syntactic_distances)
    phonological_distances_min = get_min_dist(phonological_distances)
    genetic_distances_min = get_min_dist(genetic_distances)
    geographical_distances_min = get_min_dist(geographical_distances)
    inventory_distances_min = get_min_dist(inventory_distances)

    # Dump top-1 minimum distances to csv
    with open(args.output_path + f"/{args.dataset_name}_min_distances_min.tsv", "w") as fp:
        for source_language in syntactic_distances_min.keys():
            fp.write(f"{source_language}\t{genetic_distances_min[source_language][0]}\t{genetic_distances_min[source_language][1]}\t")
            fp.write(f"{source_language}\t{geographical_distances_min[source_language][0]}\t{geographical_distances_min[source_language][1]}\t")
            fp.write(f"{source_language}\t{inventory_distances_min[source_language][0]}\t{inventory_distances_min[source_language][1]}\t")
            fp.write(f"{source_language}\t{phonological_distances_min[source_language][0]}\t{phonological_distances_min[source_language][1]}\t")
            fp.write(f"{source_language}\t{syntactic_distances_min[source_language][0]}\t{syntactic_distances_min[source_language][1]}\n")
    
    fp.close()

if __name__ == "__main__":
    main()