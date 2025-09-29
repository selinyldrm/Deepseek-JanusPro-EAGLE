def average_generation_time(file_path):
    times = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("generate time="):
                try:
                    # Extract the number between "time=" and "seconds"
                    parts = line.split("=")[1].split()[0]
                    time_val = float(parts)
                    times.append(time_val)
                except (IndexError, ValueError):
                    continue  # skip malformed lines

    if not times:
        return None

    avg_time = sum(times) / len(times)
    return avg_time, len(times)


# Example usage:
if __name__ == "__main__":
    file_path = "/work1/deming/seliny2/LANTERN/5k-227580.log"  # replace with your filename
    result = average_generation_time(file_path)
    if result:
        avg, count = result
        print(f"Read {count} generation times")
        print(f"Average time: {avg:.6f} seconds")
    else:
        print("No valid generation times found.")
