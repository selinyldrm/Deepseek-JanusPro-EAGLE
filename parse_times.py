#!/usr/bin/env python3
import re
import sys

def parse_generation_times(filename):
    pattern = re.compile(r"=(\d+\.\d+) sec")
    times = []

    with open(filename, "r") as f:
        for line in f:
            matches = pattern.findall(line)
            if matches:
                times.extend(float(m) for m in matches)

    if not times:
        print("No generation times found.")
        return

    total = sum(times)
    avg = total / len(times)
    print(f"Found {len(times)} entries.")
    print(f"Total generation time: {total:.6f} seconds")
    print(f"Average generation time: {avg:.6f} seconds")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_log.txt>")
        sys.exit(1)
    parse_generation_times(sys.argv[1])
