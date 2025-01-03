package main

/*
#include <stdint.h>

typedef int (*CallbackFunc)(const char*);

typedef struct {
	long seed;
	int width;
	int height;
	int food_spawn_chance;
	int minimum_food;
} GameSetting;

typedef struct {
	CallbackFunc on_start;
	CallbackFunc on_move;
	CallbackFunc on_end;
} Client;

__attribute__((weak))
int callCallback(CallbackFunc cb, const char* value) {
	return cb(value);
}
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"github.com/BattlesnakeOfficial/rules"
	"github.com/BattlesnakeOfficial/rules/client"
	"github.com/BattlesnakeOfficial/rules/maps"
	"math/rand"
	"time"
)

func main() {
}

type Client struct {
	onStart C.CallbackFunc
	onMove  C.CallbackFunc
	onEnd   C.CallbackFunc
}

type GameSetting struct {
	// ゲームの設定
	Seed            int64
	Width           int
	Height          int
	FoodSpawnChance int
	MinimumFood     int
}

type SnakeState struct {
	ID       string
	Client   *Client
	LastMove string
}

type GameState struct {
	// ゲームの状態
	gameId      string
	gameSetting *GameSetting
	gameMap     maps.GameMap
	ruleSet     rules.Ruleset
	settings    map[string]string
	snakeStates map[string]SnakeState
}

type GameResult struct {
	Result string         `json:"result"`
	Cause  string         `json:"cause"`
	Turn   int            `json:"turn"`
	You    client.Snake   `json:"you"`
	Snakes []client.Snake `json:"snakes"`
}

func (c *Client) callOnStart() {
	C.callCallback(c.onStart, C.CString(""))
}

func (c *Client) callOnMove(json string) int {
	return int(C.callCallback(c.onMove, C.CString(json)))
}

func (c *Client) callOnEnd(json string) {
	C.callCallback(c.onEnd, C.CString(json))
}

func clientFromCStruct(strct C.Client) *Client {
	return &Client{
		onStart: strct.on_start,
		onMove:  strct.on_move,
		onEnd:   strct.on_end,
	}
}

func gameSettingFromCStruct(strct C.GameSetting) *GameSetting {
	return &GameSetting{
		Seed:            int64(strct.seed),
		Width:           int(strct.width),
		Height:          int(strct.height),
		FoodSpawnChance: int(strct.food_spawn_chance),
		MinimumFood:     int(strct.minimum_food),
	}
}

//export StartSoloGame
func StartSoloGame(client C.Client, setting C.GameSetting) *C.char {
	c := clientFromCStruct(client)
	s := gameSettingFromCStruct(setting)
	res, err := startGame([]*Client{c}, s)
	if err != nil {
		return C.CString("")
	}
	bytes, err := json.Marshal(res)
	if err != nil {
		return C.CString("")
	}
	return C.CString(string(bytes))
}

//export StartDuelGame
func StartDuelGame(client1 C.Client, client2 C.Client, setting C.GameSetting) *C.char {
	c1 := clientFromCStruct(client1)
	c2 := clientFromCStruct(client2)
	s := gameSettingFromCStruct(setting)
	res, err := startGame([]*Client{c1, c2}, s)
	if err != nil {
		return C.CString("")
	}
	bytes, err := json.Marshal(res)
	if err != nil {
		return C.CString("")
	}
	return C.CString(string(bytes))
}

func startGame(clients []*Client, setting *GameSetting) (*GameResult, error) {
	gameState := &GameState{gameSetting: setting}
	gameState.gameId = "game-" + time.Now().Format("20060102150405")
	gameState.snakeStates = make(map[string]SnakeState)
	for i, client := range clients {
		gameState.snakeStates["snake-"+fmt.Sprint(i)] = SnakeState{ID: "snake-" + fmt.Sprint(i), Client: client, LastMove: "up"}
	}
	gameMap, _ := maps.GetMap("standard")
	gameState.gameMap = gameMap
	gameState.settings = map[string]string{
		rules.ParamFoodSpawnChance:     fmt.Sprint(setting.FoodSpawnChance),
		rules.ParamMinimumFood:         fmt.Sprint(setting.MinimumFood),
		rules.ParamHazardDamagePerTurn: fmt.Sprint(0),
		rules.ParamShrinkEveryNTurns:   fmt.Sprint(0),
	}
	ruleSet := rules.NewRulesetBuilder().WithSeed(setting.Seed).WithParams(gameState.settings).WithSolo(len(clients) == 1).NamedRuleset("standard")
	gameState.ruleSet = ruleSet

	rand.Seed(gameState.gameSetting.Seed)

	gameOver, boardState, err := initializeBoard(gameState)
	if err != nil {
		return nil, err
	}

	for !gameOver { //メインループ
		gameOver, boardState, err = gameState.createNextBoard(boardState)
		if err != nil {
			return nil, err
		}
	}

	for _, snakeState := range gameState.snakeStates {
		snakeRequest := gameState.getRequestBodyForSnake(boardState, snakeState)
		requestBody := serialiseSnakeRequest(snakeRequest)
		snakeState.Client.callOnEnd(string(requestBody))
	}

	var youSnake *rules.Snake
	for _, snk := range boardState.Snakes {
		if "snake-0" == snk.ID {
			youSnake = &snk
			break
		}
	}

	var result string
	if youSnake.EliminatedCause == rules.NotEliminated {
		result = "win"
	} else {
		result = "lose"
	}
	if len(gameState.snakeStates) > 1 {
		var isDraw = true
		for _, snake := range boardState.Snakes {
			if snake.EliminatedCause != rules.NotEliminated {
				isDraw = false
				break
			}
		}
		if isDraw {
			result = "draw"
		}
	}

	return &GameResult{
		Result: result,
		Cause:  youSnake.EliminatedCause,
		Turn:   boardState.Turn,
		You:    convertRulesSnake(*youSnake, gameState.snakeStates[youSnake.ID]),
		Snakes: convertRulesSnakes(boardState.Snakes, gameState.snakeStates, true),
	}, nil

}

func initializeBoard(gameState *GameState) (bool, *rules.BoardState, error) {
	// 盤面を初期化する
	snakeIds := []string{}
	for _, snakeState := range gameState.snakeStates {
		snakeIds = append(snakeIds, snakeState.ID)
	}
	boardState, err := maps.SetupBoard(gameState.gameMap.ID(), gameState.ruleSet.Settings(), gameState.gameSetting.Width, gameState.gameSetting.Height, snakeIds)
	if err != nil {
		return false, nil, err
	}

	gameOver, boardState, err := gameState.ruleSet.Execute(boardState, nil)
	if err != nil {
		return false, nil, err
	}
	for _, snakeState := range gameState.snakeStates {
		snakeState.Client.callOnStart()
	}
	return gameOver, boardState, nil
}

func (gameState *GameState) createNextBoard(boardState *rules.BoardState) (bool, *rules.BoardState, error) {
	boardState, err := maps.PreUpdateBoard(gameState.gameMap, boardState, gameState.ruleSet.Settings())
	if err != nil {
		return false, nil, err
	}
	var moves []rules.SnakeMove
	for _, snakeState := range gameState.snakeStates {
		for _, snake := range boardState.Snakes {
			if snake.ID == snakeState.ID && snake.EliminatedCause == rules.NotEliminated {
				nextState := gameState.getSnakeUpdate(boardState, snakeState)
				gameState.snakeStates[snakeState.ID] = nextState
				moves = append(moves, rules.SnakeMove{ID: snakeState.ID, Move: nextState.LastMove})
			}
		}
	}

	gameOver, boardState, err := gameState.ruleSet.Execute(boardState, moves)
	if err != nil {
		return false, nil, err
	}

	boardState, err = maps.PostUpdateBoard(gameState.gameMap, boardState, gameState.ruleSet.Settings())
	if err != nil {
		return false, nil, err
	}
	boardState.Turn++
	return gameOver, boardState, nil
}

func (gameState *GameState) getSnakeUpdate(boardState *rules.BoardState, snakeState SnakeState) SnakeState {
	snakeReq := gameState.getRequestBodyForSnake(boardState, snakeState)
	reqBody := serialiseSnakeRequest(snakeReq)
	resBody := snakeState.Client.callOnMove(string(reqBody))
	switch resBody {
	case 0:
		snakeState.LastMove = "up"
	case 1:
		snakeState.LastMove = "down"
	case 2:
		snakeState.LastMove = "left"
	case 3:
		snakeState.LastMove = "right"
	default:
		snakeState.LastMove = "up"
	}
	return snakeState
}

func serialiseSnakeRequest(snakeRequest client.SnakeRequest) []byte {
	requestJSON, _ := json.Marshal(snakeRequest)
	return requestJSON
}

func (gameState *GameState) getRequestBodyForSnake(boardState *rules.BoardState, snakeState SnakeState) client.SnakeRequest {
	var youSnake rules.Snake
	for _, snk := range boardState.Snakes {
		if snakeState.ID == snk.ID {
			youSnake = snk
			break
		}
	}
	request := client.SnakeRequest{
		Game:  gameState.createClientGame(),
		Turn:  boardState.Turn,
		Board: convertStateToBoard(boardState, gameState.snakeStates),
		You:   convertRulesSnake(youSnake, snakeState),
	}
	return request
}

func (gameState *GameState) createClientGame() client.Game {
	return client.Game{
		ID:      gameState.gameId,
		Timeout: 500,
		Ruleset: client.Ruleset{
			Name:     gameState.ruleSet.Name(),
			Version:  "cli",
			Settings: client.ConvertRulesetSettings(gameState.ruleSet.Settings()),
		},
		Map: gameState.gameMap.ID(),
	}
}

func convertRulesSnake(snake rules.Snake, snakeState SnakeState) client.Snake {
	return client.Snake{
		ID:      snake.ID,
		Name:    snakeState.ID,
		Health:  snake.Health,
		Body:    client.CoordFromPointArray(snake.Body),
		Latency: "0",
		Head:    client.CoordFromPoint(snake.Body[0]),
		Length:  len(snake.Body),
		Shout:   "",
		Customizations: client.Customizations{
			Head:  "",
			Tail:  "",
			Color: "",
		},
	}
}

func convertRulesSnakes(snakes []rules.Snake, snakeStates map[string]SnakeState, includeEliminated bool) []client.Snake {
	a := make([]client.Snake, 0)
	for _, snake := range snakes {
		if snake.EliminatedCause == rules.NotEliminated || includeEliminated {
			a = append(a, convertRulesSnake(snake, snakeStates[snake.ID]))
		}
	}
	return a
}

func convertStateToBoard(boardState *rules.BoardState, snakeStates map[string]SnakeState) client.Board {
	return client.Board{
		Height:  boardState.Height,
		Width:   boardState.Width,
		Food:    client.CoordFromPointArray(boardState.Food),
		Hazards: client.CoordFromPointArray(boardState.Hazards),
		Snakes:  convertRulesSnakes(boardState.Snakes, snakeStates, false),
	}
}
