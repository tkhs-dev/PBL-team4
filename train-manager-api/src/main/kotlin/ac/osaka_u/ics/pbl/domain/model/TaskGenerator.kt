package ac.osaka_u.ics.pbl.domain.model

import ac.osaka_u.ics.pbl.common.LeaderboardApi
import ac.osaka_u.ics.pbl.common.TaskGeneratorType
import ac.osaka_u.ics.pbl.common.TaskStatus
import ac.osaka_u.ics.pbl.common.TaskType
import ac.osaka_u.ics.pbl.domain.repos.ModelRepository
import ac.osaka_u.ics.pbl.domain.repos.TaskRepository
import kotlinx.datetime.Clock
import java.util.*

data class TaskGenerator(
    val id: Int,
    val name: String,
    val type: TaskGeneratorType,
    val weight: Int,
    val parameters: Map<String,Any>
)

fun TaskGenerator.generateTask(taskRepos: TaskRepository, modelRepository: ModelRepository): Task? {
    val taskType = when(type){
        TaskGeneratorType.SPECIFIC_PLAYER -> TaskType.SUPERVISED
        TaskGeneratorType.RANDOM_MATCH -> TaskType.SUPERVISED
    }
    val newestTask = taskRepos.findNewestByGeneratorId(id, taskType)
    val baseModel = newestTask?.let{
        when(it.status){
            TaskStatus.WAITING -> return null
            TaskStatus.PROCESSING -> return null
            TaskStatus.ERROR -> {
                it.baseModelId?.let { baseModelId ->
                    modelRepository.findModelById(baseModelId)
                }
            }
            TaskStatus.COMPLETED -> {
                modelRepository.findModelByTaskId(it.id)
            }
        }
    }
    val parameter = when(type){
        TaskGeneratorType.SPECIFIC_PLAYER -> {
            val playerId = parameters["player_id"] as String
            val games = LeaderboardApi.getPlayerGames(playerId).shuffled().take(parameters["game_count"] as Int).map { it.gameId }
            if (games.isEmpty()) return null
            mapOf(
                "games" to games.map { "${playerId}_${it}"},
                "epochs" to (parameters["epochs"] ?: 200) as Int,
            )
        }
        TaskGeneratorType.RANDOM_MATCH -> {
            val games = LeaderboardApi.getLeaderboard(limit = 10)
                .map { it to LeaderboardApi.getPlayerGames(it).filter { it.turns >= 50 } }
                .flatMap { (player, games) -> games.map { "${player}_${it.gameId}" } }
            mapOf(
                "games" to games.shuffled().take(parameters["game_count"] as Int),
                "epochs" to (parameters["epochs"] ?: 200) as Int,
            )
        }
    }
    return taskRepos.createTask(
        Task(
            id = UUID.randomUUID(),
            status = TaskStatus.WAITING,
            errorCount = 0,
            baseModelId = baseModel?.id,
            type = taskType,
            createdAt = Clock.System.now(),
            generatorId = id,
            parameter = parameter,
        )
    )
}

fun TaskGenerator.validateParameters(){
    when(type){
        TaskGeneratorType.SPECIFIC_PLAYER -> {
            if(parameters["player_id"]?.let { LeaderboardApi.playerExists(it as String) } == false){
                throw IllegalArgumentException("valid player_id is required for supervised task")
            }
            if(parameters["game_count"]?.let { it is Int } == false){
                throw IllegalArgumentException("game_count is required for supervised task")
            }
        }
        TaskGeneratorType.RANDOM_MATCH -> {
            if(parameters["game_count"]?.let { it is Int } == false){
                throw IllegalArgumentException("game_count is required for supervised task")
            }
        }
    }
}