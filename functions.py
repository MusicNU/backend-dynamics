
def add_five(num: int) -> int:
    return num + 5

def divider(dividend: int, divider: int) -> float:
    if divider == 0:
        return float('inf')
    return dividend / divider