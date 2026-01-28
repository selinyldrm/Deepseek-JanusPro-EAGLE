import re

def average_generate_time(file_path: str) -> float:
    pattern = re.compile(r"generate time=([\d.]+)\s*seconds")
    times = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                times.append(float(match.group(1)))

    if not times:
        raise ValueError("No generate time entries found in the file.")

    return sum(times) / len(times)


if __name__ == "__main__":
    file_path = "/work1/deming/seliny2/LANTERN/lumina-264128.log"  # replace with your file path
    avg_time = average_generate_time(file_path)
    print(f"Average generate time: {avg_time:.6f} seconds")
