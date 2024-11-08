import json
import unittest
from unittest import TestCase

from shared.parameter_util import get_snake_foods, get_front_body, get_snake_health, get_snake_length, \
    get_snake_distance, get_right_body, get_left_body, get_leftd_body, get_rightd_body

# test json
json_text = """
{"game":{"id":"7bec30be-e3e3-4fad-98b9-907a3921cbfc","ruleset":{"name":"solo","version":"cli","settings":{"foodSpawnChance":15,"minimumFood":1,"hazardDamagePerTurn":14,"hazardMap":"","hazardMapAuthor":"","royale":{"shrinkEveryNTurns":25},"squad":{"allowBodyCollisions":false,"sharedElimination":false,"sharedHealth":false,"sharedLength":false}}},"map":"standard","timeout":500,"source":""},"turn":28,"board":{"height":6,"width":6,"snakes":[{"id":"db326dc5-feb4-4a8a-89af-940d49a067f4","name":"'Local","latency":"306","health":98,"body":[{"x":2,"y":0},{"x":2,"y":1},{"x":1,"y":1},{"x":0,"y":1},{"x":0,"y":0}],"head":{"x":2,"y":0},"length":5,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}}],"food":[{"x":3,"y":0},{"x":4,"y":1},{"x":3,"y":5}],"hazards":[]},"you":{"id":"db326dc5-feb4-4a8a-89af-940d49a067f4","name":"'Local","latency":"0","health":98,"body":[{"x":2,"y":0},{"x":2,"y":1},{"x":1,"y":1},{"x":0,"y":1},{"x":0,"y":0}],"head":{"x":2,"y":0},"length":5,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}}}
"""
json_dict = json.loads(json_text)

json_text2 = """
{"game":{"id":"7bec30be-e3e3-4fad-98b9-907a3921cbfc","ruleset":{"name":"solo","version":"cli","settings":{"foodSpawnChance":15,"minimumFood":1,"hazardDamagePerTurn":14,"hazardMap":"","hazardMapAuthor":"","royale":{"shrinkEveryNTurns":25},"squad":{"allowBodyCollisions":false,"sharedElimination":false,"sharedHealth":false,"sharedLength":false}}},"map":"standard","timeout":500,"source":""},"turn":28,"board":{"height":6,"width":6,"snakes":[{"id":"db326dc5-feb4-4a8a-89af-940d49a067f4","name":"'Local","latency":"306","health":98,"body":[{"x":2,"y":0},{"x":2,"y":1},{"x":1,"y":1},{"x":0,"y":1},{"x":0,"y":0}],"head":{"x":2,"y":0},"length":5,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}}],"food":[{"x":0,"y":1}],"hazards":[]},"you":{"id":"db326dc5-feb4-4a8a-89af-940d49a067f4","name":"'Local","latency":"0","health":98,"body":[{"x":1,"y":3},{"x":0,"y":3},{"x":0,"y":4},{"x":1,"y":4},{"x":1,"y":5}],"head":{"x":1,"y":3},"length":5,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}}}
"""
json_dict2 = json.loads(json_text2)

if __name__ == '__main__':
    unittest.main()


class Test(TestCase):
    def test_get_front_body(self):
        expect = 0
        actual = get_front_body(json_dict)
        self.assertEqual(expect, actual)

    def test_get_right_body(self):
        expect = 1
        actual = get_right_body(json_dict)
        self.assertEqual(expect, actual)

    def test_get_rightd_body(self):
        expect = 0
        actual = get_rightd_body(json_dict)
        self.assertEqual(expect, actual)

    def test_get_left_body(self):
        expect = 0
        actual = get_left_body(json_dict)
        self.assertEqual(expect, actual)

    def test_get_leftd_body(self):
        expect = 0
        actual = get_leftd_body(json_dict)
        self.assertEqual(expect, actual)

    def test_get_snake_foods(self):
        expect = 1
        actual = get_snake_foods(json_dict)
        self.assertEqual(expect, actual)
        print("!!")
        expect = 3
        actual = get_snake_foods(json_dict2)
        self.assertEqual(expect, actual)

    def test_get_snake_distance(self):
        expect = 0
        actual = get_snake_distance(json_dict)
        self.assertEqual(expect, actual)

    def test_get_snake_length(self):
        expect = 5
        actual = get_snake_length(json_dict)
        self.assertEqual(expect, actual)

    def test_get_snake_health(self):
        expect = 98
        actual = get_snake_health(json_dict)
        self.assertEqual(expect, actual)
