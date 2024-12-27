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
    return when(type){
        TaskGeneratorType.SPECIFIC_PLAYER -> {
            val playerId = parameters["player_id"] as String
            val rootModelId = parameters["root_model_id"] as UUID
            val newestTask = taskRepos.findNewestSupervisedTaskByRoot(rootModelId.toString())
            val model = if (newestTask == null){
                modelRepository.findModelById(rootModelId) ?: return null
            }else{
                when(newestTask.status){
                    TaskStatus.WAITING -> return newestTask
                    TaskStatus.PROCESSING -> {return null}
                    TaskStatus.ERROR -> {
                        newestTask.baseModelId?.let {
                            modelRepository.findModelById(it)
                        }
                    }
                    TaskStatus.COMPLETED -> {
                        modelRepository.findModelByTaskId(newestTask.id)
                    }
                }
            }

            val games = LeaderboardApi.getPlayerGames(playerId).shuffled().take(parameters["game_count"] as Int).map { it.gameId }
            if (games.isEmpty()) {
                return null
            }
            val epochs = (parameters["epochs"] ?: 200) as Int
            val parameter = mapOf(
                "games" to games,
                "epochs" to epochs,
            )
            val task = Task(
                id = UUID.randomUUID(),
                status = TaskStatus.WAITING,
                errorCount = 0,
                baseModelId = model?.id,
                type = TaskType.SUPERVISED,
                createdAt = Clock.System.now(),
                parameter = parameter,
            )
            return taskRepos.createTask(task)
        }
        TaskGeneratorType.RANDOM_MATCH -> {
            null
        }
    }
}

fun TaskGenerator.validateParameters(){
    when(type){
        TaskGeneratorType.SPECIFIC_PLAYER -> {
            if(parameters["root_model_id"] == null){
                throw IllegalArgumentException("root_model_id is required for supervised task")
            }
            if(parameters["player_id"] == null){
                throw IllegalArgumentException("player_id is required for supervised task")
            }
            if(parameters["game_count"]?.let { it is Int } == false){
                throw IllegalArgumentException("game_count is required for supervised task")
            }
        }
        TaskGeneratorType.RANDOM_MATCH -> {
            // do nothing
        }
    }
}