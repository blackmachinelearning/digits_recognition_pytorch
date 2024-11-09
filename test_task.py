# Есть строка со случайными символами. Надо посчитать, каких символов больше — в случае если 
# в нижнем регистре, то вывод должен быть -1, в случае если в верхнем регистре, то вывод должен быть 1, 0 если равно
import pytest

def check_register(string: str):
    # upper_count = len(re.sub(pattern=r"[A-Z]"))
    # lower_count = len(re.sub(pattern=r"[a-z]"))
    upper_count = 0
    lower_count = 0

    for s in string:
        if s.isupper():
            upper_count += 1
        elif s.islower():
            lower_count +=1

    if upper_count > lower_count:
        return 1
    elif lower_count > upper_count:
        return -1
    else:
        return 0

@pytest.mark.parametrize(
        "string,result",
        [
            ("", 0),
            ("AAAA", 1),
            ("aaa", -1),
            ("Aa", 0),
            ("aA", 0),
            ("AAaaA", 1),
            ("1232", 0),
            ("&(*)", 0),
        ]
)
def test_check_register(string, result):
    assert check_register(string=string) == result

# @pytest.mark.parametrize(
#         "string,result",
#         [
#             ([""], 0),
#             (12, 1),
#             (-1, -1),
#             (3 +3j, 0),
#             (True, 0),
#             (False, 1),
#             (1.1, 0),
#             (None, 0),
#         ]
# )
# def test_check_register(string, result):
#     assert check_register(string=string) == result


# ""
# "AAAA"
# "aaa"
# "Aa"
# "aA"
# "AAaaA"
# "1232"
# "&(*)"