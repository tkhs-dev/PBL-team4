package ac.osaka_u.ics.pbl.common

import kotlinx.coroutines.runBlocking
import org.jsoup.Jsoup

class LeaderboardApi {
    companion object {
        const val BASE_URL = "https://play.battlesnake.com/leaderboard/standard-duels"
        fun getLeaderboard(limit: Int = 100): List<String> {
            return runBlocking {
                val doc = Jsoup.connect(BASE_URL).get()
                val table = doc.selectXpath("""/html/body/div[1]/div/main/div[2]/div[2]/table/tbody/tr""")
                table.drop(1).take(limit).map {
                    it.select("td")[2].text()
                }
            }
        }

        fun playerExists(playerId: String): Boolean {
            return runBlocking {
                try {
                    Jsoup.connect("$BASE_URL/$playerId/stats").get()
                    true
                } catch (e: Exception) {
                    false
                }
            }
        }

        fun getPlayerGames(playerId: String, limit: Int = 100, resultQuery: Set<GameResult> = setOf(GameResult.WIN)): List<PlayerGame> {
            return runBlocking {
                val doc = try{
                    Jsoup.connect("$BASE_URL/$playerId/stats").get()
                }catch (e: Exception){
                    return@runBlocking emptyList<PlayerGame>()
                }
                val table = doc.selectXpath("""/html/body/div[1]/div/main/div[2]/table/tbody/tr""")
                table.drop(0).map {
                    val tds = it.select("td")
                    val resultTxt = tds[0].text()
                    val result = when {
                        resultTxt.contains("Winner") -> GameResult.WIN
                        resultTxt.contains("Eliminated") -> GameResult.LOSS
                        else -> GameResult.DRAW
                    }
                    val turns = tds[2].text().toInt()
                    val gameId = tds[3].select("a").attr("href").drop(6)
                    PlayerGame(gameId, result, turns)
                }.filter { it.result in resultQuery }
                    .take(limit)
            }
        }
    }

    enum class GameResult {
        WIN, LOSS, DRAW
    }

    data class PlayerGame(val gameId: String, val result: GameResult, val turns: Int)
}