package ac.osaka_u.ics.pbl.handler

import ac.osaka_u.ics.pbl.common.TaskStatus
import ac.osaka_u.ics.pbl.domain.model.Task
import ac.osaka_u.ics.pbl.domain.repos.QueueRepository
import ac.osaka_u.ics.pbl.domain.repos.TaskRepository
import ac.osaka_u.ics.pbl.model.TaskRequest
import ac.osaka_u.ics.pbl.model.TaskResponse
import ac.osaka_u.ics.pbl.model.toResponse
import io.ktor.server.plugins.*
import kotlinx.datetime.Clock
import java.util.*

class QueueHandler(private val queueRepository: QueueRepository, private val taskRepository: TaskRepository) {
    fun handleGetAll(): List<TaskResponse> {
        return queueRepository.get().map { it.toResponse() }
    }

    fun handlePost(request:TaskRequest): List<TaskResponse> {
        val baseUid = try {
            request.baseModelId?.let { UUID.fromString(it) }
        } catch (e: IllegalArgumentException) {
            throw BadRequestException("Invalid baseModelId")
        }
        baseUid?.let {
            if (taskRepository.findTaskById(it) == null) {
                throw BadRequestException("Invalid baseModelId")
            }
        }

        val task = taskRepository.createTask(
            Task(
                id = UUID.randomUUID(),
                status = TaskStatus.WAITING,
                errorCount = 0,
                baseModelId = baseUid,
                type = request.type,
                createdAt = Clock.System.now(),
                generatorId = request.generatorReference,
                parameter = request.parameters
            )
        )
        queueRepository.enqueue(task.id)
        return handleGetAll()
    }
}