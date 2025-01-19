from functions import add_five, divider

def test_add_five_default() -> None:
    assert add_five(100) == 105
    assert add_five(-45) == -40

def test_divider_default() -> None:
    assert divider(35, 9) == (35 / 9)
    assert divider(10, 10) == .4
    assert divider(-34, 0) == float('inf')