import re

def average_generate_time(file_path: str) -> float:
    # Flexible pattern: matches 'generate time', optional colon/equals, then the number
    pattern = re.compile(r"generate time[:\s=]*([\d.]+)")
    times = []

    with open(file_path, "r") as f:
        for line in f:
            # Case-insensitive search for better coverage
            match = pattern.search(line.lower())
            if match:
                try:
                    val = float(match.group(1))
                    times.append(val)
                except ValueError:
                    continue
            # DEBUG: If 'generate time' is in the line but it didn't match, show us why
            elif "generate time" in line.lower():
                print(f"Potential match failed on line: {line.strip()}")

    if not times:
        print("\n--- ERROR ---")
        print(f"Total lines searched: {sum(1 for _ in open(file_path))}")
        raise ValueError("No generate time entries found in the file. Check the log format above.")

    return sum(times) / len(times), len(times)

if __name__ == "__main__":
    file_path = "/work1/deming/shared/lumina/inter-only-0.625-kl4.0/0-1000.txt"
    avg_time, time_length = average_generate_time(file_path)
    print(f"\nAverage generate time: {avg_time:.6f} seconds")
    print(f"\t generation count{time_length:.6f} images")