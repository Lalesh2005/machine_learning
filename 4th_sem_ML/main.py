import sys
from functools import reduce

def process_cases(n, get_line, results):
    if n == 0:
        return results
    x = int(get_line())
    nums = list(map(int, get_line().split()))
    if len(nums) != x:
        return process_cases(n - 1, get_line, results + [-1])
    non_positive = list(filter(lambda y: y <= 0, nums))
    total = reduce(lambda acc, y: acc + y ** 4, non_positive, 0)
    return process_cases(n - 1, get_line, results + [total])


def main():
    lines = iter(sys.stdin.read().splitlines())
    get_line = lambda: next(lines)
    n = int(get_line())
    results = process_cases(n, get_line, [])
    sys.stdout.write("\n".join(map(str, results)) + "\n")


if __name__ == "__main__":
    main()