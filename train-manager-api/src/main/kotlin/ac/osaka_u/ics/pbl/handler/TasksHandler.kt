package ac.osaka_u.ics.pbl.handler

import ac.osaka_u.ics.pbl.ApiException
import ac.osaka_u.ics.pbl.common.LeaderboardApi
import ac.osaka_u.ics.pbl.common.TaskType
import ac.osaka_u.ics.pbl.domain.model.TaskGenerator
import ac.osaka_u.ics.pbl.domain.repos.TaskGeneratorRepository
import ac.osaka_u.ics.pbl.domain.repos.TaskRepository
import ac.osaka_u.ics.pbl.model.PostGeneratorRequest
import ac.osaka_u.ics.pbl.model.TaskGeneratorResponse
import ac.osaka_u.ics.pbl.model.TaskResponse
import ac.osaka_u.ics.pbl.model.toResponse
import io.ktor.util.reflect.*
import java.util.*

class TasksHandler(private val taskRepo: TaskRepository, private val generatorRepo: TaskGeneratorRepository) {
    fun handleGetTasks(): List<TaskResponse> {
        return taskRepo.findTasks().map { it.toResponse() }
    }

    fun handleGetTask(id: String): TaskResponse {
        val uid = try{
            UUID.fromString(id) ?: throw IllegalArgumentException()
        }catch (e: IllegalArgumentException){
            throw ApiException.BadRequestException("Invalid task ID")
        }
        return taskRepo.findTaskById(uid)?.toResponse() ?: throw ApiException.NotFoundException()
    }

    fun handleGetGenerators(): List<TaskGeneratorResponse> {
        return generatorRepo.findTaskGenerators().map { it.toResponse() }
    }

    fun handleGetGenerator(id: Int): TaskGeneratorResponse {
        return generatorRepo.findTaskGeneratorById(id)?.toResponse() ?: throw ApiException.NotFoundException()
    }

    fun handlePostGenerator(request: PostGeneratorRequest): TaskGeneratorResponse {
        when(request.type){
            TaskType.SUPERVISED -> {
                if(request.parameters["player_id"] == null){
                    throw ApiException.BadRequestException("player_id is required for supervised task")
                }
                if(!LeaderboardApi.playerExists(request.parameters["player_id"] as String)){
                    throw ApiException.BadRequestException("specified player does not exist on the leaderboard")
                }
                if(request.parameters["game_count"]?.let { it is Int } == false){
                    throw ApiException.BadRequestException("game_count is required for supervised task")
                }
            }
            TaskType.REINFORCEMENT -> {
                // do nothing
            }
            else -> {
                throw ApiException.BadRequestException("Invalid task type")
            }
        }

        val generator = TaskGenerator(
            id = 0,
            name = request.name,
            type = request.type,
            weight = request.weight,
            parameters = request.parameters
        )
        return generatorRepo.create(generator).toResponse()
    }

    fun handleDeleteGenerator(id: Int) {
        generatorRepo.delete(id)
    }
}