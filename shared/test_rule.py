import unittest
import json

from shared import rule


class MyTestCase(unittest.TestCase):
    tr_json_text = """{"turn":0,"board":{"height":6,"width":6,"snakes":[{"id":"67f893d8-44e6-49d9-9e5a-e972045d52c1","name":"'Local","latency":"0","health":50,"body":[{"x":5,"y":5},{"x":5,"y":4},{"x":5,"y":3}],"head":{"x":5,"y":5},"length":3,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}}],"food":[{"x":0,"y":0}],"hazards":[]},"you":{"id":"67f893d8-44e6-49d9-9e5a-e972045d52c1","name":"'Local","latency":"0","health":50,"body":[{"x":5,"y":5},{"x":5,"y":4},{"x":5,"y":3}],"head":{"x":5,"y":5},"length":3,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}}}"""
    bl_json_text = """{"turn":0,"board":{"height":6,"width":6,"snakes":[{"id":"67f893d8-44e6-49d9-9e5a-e972045d52c1","name":"'Local","latency":"0","health":50,"body":[{"x":0,"y":0},{"x":0,"y":1},{"x":0,"y":2}],"head":{"x":0,"y":0},"length":3,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}}],"food":[{"x":1,"y":0}],"hazards":[]},"you":{"id":"67f893d8-44e6-49d9-9e5a-e972045d52c1","name":"'Local","latency":"0","health":50,"body":[{"x":0,"y":0},{"x":0,"y":1},{"x":0,"y":2}],"head":{"x":0,"y":0},"length":3,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}}}"""
    sl_json_text = """{"turn":0,"board":{"height":6,"width":6,"snakes":[{"id":"67f893d8-44e6-49d9-9e5a-e972045d52c1","name":"'Local","latency":"0","health":50,"body":[{"x":2,"y":2},{"x":2,"y":1},{"x":2,"y":0}],"head":{"x":2,"y":2},"length":3,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}},{"id":"67f893d8-44e6-49d9-9e5a-e972045d52c1","name":"'Local","latency":"0","health":50,"body":[{"x":2,"y":3},{"x":1,"y":3},{"x":1,"y":2}],"head":{"x":2,"y":3},"length":3,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}}],"food":[{"x":0,"y":0}],"hazards":[]},"you":{"id":"67f893d8-44e6-49d9-9e5a-e972045d52c1","name":"'Local","latency":"0","health":50,"body":[{"x":2,"y":2},{"x":2,"y":1},{"x":2,"y":0}],"head":{"x":2,"y":2},"length":3,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}}}"""
    s_json_text = """{"turn":0,"board":{"height":6,"width":6,"snakes":[{"id":"67f893d8-44e6-49d9-9e5a-e972045d52c1","name":"'Local","latency":"0","health":50,"body":[{"x":2,"y":2},{"x":2,"y":1},{"x":2,"y":0}],"head":{"x":2,"y":2},"length":3,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}},{"id":"67f893d8-44e6-49d9-9e5a-e972045d52c1","name":"'Local","latency":"0","health":50,"body":[{"x":2,"y":3},{"x":1,"y":3}],"head":{"x":2,"y":3},"length":2,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}}],"food":[{"x":0,"y":0}],"hazards":[]},"you":{"id":"67f893d8-44e6-49d9-9e5a-e972045d52c1","name":"'Local","latency":"0","health":50,"body":[{"x":2,"y":2},{"x":2,"y":1},{"x":2,"y":0}],"head":{"x":2,"y":2},"length":3,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}}}"""
    l_json_text = """{"turn":0,"board":{"height":6,"width":6,"snakes":[{"id":"67f893d8-44e6-49d9-9e5a-e972045d52c1","name":"'Local","latency":"0","health":50,"body":[{"x":2,"y":2},{"x":2,"y":1},{"x":2,"y":0}],"head":{"x":2,"y":2},"length":3,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}},{"id":"67f893d8-44e6-49d9-9e5a-e972045d52c1","name":"'Local","latency":"0","health":50,"body":[{"x":2,"y":3},{"x":1,"y":3},{"x":1,"y":2},{"x":1,"y":3}],"head":{"x":2,"y":3},"length":4,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}}],"food":[{"x":0,"y":0}],"hazards":[]},"you":{"id":"67f893d8-44e6-49d9-9e5a-e972045d52c1","name":"'Local","latency":"0","health":50,"body":[{"x":2,"y":2},{"x":2,"y":1},{"x":2,"y":0}],"head":{"x":2,"y":2},"length":3,"shout":"","squad":"","customizations":{"color":"#888888","head":"default","tail":"default"}}}"""
    tr_json = json.loads(tr_json_text)
    bl_json = json.loads(bl_json_text)
    sl_json = json.loads(sl_json_text)
    s_json = json.loads(s_json_text)
    l_json = json.loads(l_json_text)

    def test_out_of_bounds(self):
        result = rule.move(self.tr_json, rule.Direction.UP)
        assert result[0] == rule.TurnResult.LOSE
        assert result[1] is None
        result = rule.move(self.tr_json, rule.Direction.RIGHT)
        assert result[0] == rule.TurnResult.LOSE
        assert result[1] is None
        result = rule.move(self.bl_json, rule.Direction.DOWN)
        assert result[0] == rule.TurnResult.LOSE
        assert result[1] is None
        result = rule.move(self.bl_json, rule.Direction.LEFT)
        assert result[0] == rule.TurnResult.LOSE
        assert result[1] is None

    def test_colliding_with_self(self):
        result = rule.move(self.tr_json, rule.Direction.DOWN)
        assert result[0] == rule.TurnResult.LOSE
        assert result[1] is None

    def test_normal_move(self):
        result = rule.move(self.tr_json, rule.Direction.LEFT)
        assert result[0] == rule.TurnResult.CONTINUE
        assert result[1]["you"]["length"] == 3
        assert result[1]["you"]["body"][0] == {"x": 4, "y": 5}
        assert result[1]["you"]["body"][1] == {"x": 5, "y": 5}
        assert result[1]["you"]["body"][2] == {"x": 5, "y": 4}
        assert result[1]["you"]["head"] == {"x": 4, "y": 5}
        assert result[1]["you"]["health"] == 50

    def test_colliding_with_food(self):
        result = rule.move(self.bl_json, rule.Direction.RIGHT)
        assert result[0] == rule.TurnResult.CONTINUE
        assert result[1]["you"]["length"] == 4
        assert result[1]["you"]["body"][0] == {"x": 1, "y": 0}
        assert result[1]["you"]["body"][1] == {"x": 0, "y": 0}
        assert result[1]["you"]["body"][2] == {"x": 0, "y": 1}
        assert result[1]["you"]["body"][3] == {"x": 0, "y": 2}
        assert result[1]["you"]["head"] == {"x": 1, "y": 0}
        assert result[1]["you"]["health"] == 100

    def test_colliding_with_other_snake(self):
        result = rule.move(self.sl_json, rule.Direction.UP)
        assert result[0] == rule.TurnResult.DRAW
        assert result[1] is None
        result = rule.move(self.sl_json, rule.Direction.LEFT)
        assert result[0] == rule.TurnResult.LOSE
        assert result[1] is None
        result = rule.move(self.s_json, rule.Direction.UP)
        assert result[0] == rule.TurnResult.WIN
        assert result[1] is None
        result = rule.move(self.s_json, rule.Direction.LEFT)
        result = rule.move(result[1], rule.Direction.UP)
        assert result[0] == rule.TurnResult.WIN
        assert result[1] is None
        result = rule.move(self.l_json, rule.Direction.UP)
        assert result[0] == rule.TurnResult.LOSE
        assert result[1] is None
        result = rule.move(self.l_json, rule.Direction.LEFT)
        assert result[0] == rule.TurnResult.LOSE
        assert result[1] is None





if __name__ == '__main__':
    unittest.main()
