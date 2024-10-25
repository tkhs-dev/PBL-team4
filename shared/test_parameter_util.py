import unittest
import json
from unittest import TestCase

from shared.parameter_util import get_snake_foods, get_up_body, get_snake_health, get_snake_length, get_snake_distance, \
    get_down_body, get_right_body, get_left_body

# test json
json_text = """
{"game":{"id":"7bec30be-e3e3-4fad-98b9-907a3921cbfc","ruleset":{"name":"solo","version":"cli","settings":{"foodSpawnChance":15,"minimumFood":1,"hazardDamagePerTurn":14,"hazardMap":"","hazardMapAuthor":"","royale":{"shrinkEveryNTurns":25},"squad":{"allowBodyCollisions":false,"sharedElimination":false,"sharedHealth":false,"sharedLength":false}}},"map":"standard","timeout":500,"source":""},"turn":28,"board":{"height":6,"width":6,"snakes":[{"id":"db326dc5-feb4-4a8a-89af-940d49a067f4","name":"'Local","latency":"306","health":98,"body":[{"x":2,"y":0},{"x":2,"y":1},{"x":1,"y":1},{"x":0,"y":1},{"x":0,"y":0}],"head":{"x":2,"y":0},"length":5,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}}],"food":[{"x":3,"y":0},{"x":4,"y":1},{"x":3,"y":5}],"hazards":[]},"you":{"id":"db326dc5-feb4-4a8a-89af-940d49a067f4","name":"'Local","latency":"0","health":98,"body":[{"x":2,"y":0},{"x":2,"y":1},{"x":1,"y":1},{"x":0,"y":1},{"x":0,"y":0}],"head":{"x":2,"y":0},"length":5,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}}}
"""
json = json.loads(json_text)

if __name__ == '__main__':
    unittest.main()


class Test(TestCase):
    def test_get_up_body(self):
        expect = 0
        actual = get_up_body(json)
        self.assertEqual(expect, actual)

    def test_get_down_body(self):
        expect = 0
        actual = get_down_body(json)
        self.assertEqual(expect, actual)

    def test_get_right_body(self):
        expect = 1
        actual = get_right_body(json)
        self.assertEqual(expect, actual)

    def test_get_left_body(self):
        expect = 0
        actual = get_left_body(json)
        self.assertEqual(expect, actual)

    def test_get_snake_foods(self):
        expect = 3
        actual = get_snake_foods(json)
        self.assertEqual(expect, actual)

    def test_get_snake_distance(self):
        expect = 0
        actual = get_snake_distance(json)
        self.assertEqual(expect, actual)

    def test_get_snake_length(self):
        expect = 5
        actual = get_snake_length(json)
        self.assertEqual(expect, actual)

    def test_get_snake_health(self):
        expect = 98
        actual = get_snake_health(json)
        self.assertEqual(expect, actual)
