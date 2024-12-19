package ac.osaka_u.ics.pbl.handler

import ac.osaka_u.ics.pbl.ApiException
import ac.osaka_u.ics.pbl.common.AssignmentStatus
import ac.osaka_u.ics.pbl.domain.model.Assignment
import ac.osaka_u.ics.pbl.domain.repos.AssignmentRepository
import ac.osaka_u.ics.pbl.domain.repos.QueueRepository
import ac.osaka_u.ics.pbl.generateTask
import ac.osaka_u.ics.pbl.model.Task
import ac.osaka_u.ics.pbl.model.TaskGeneratorResponse
import ac.osaka_u.ics.pbl.model.TaskResponse
import ac.osaka_u.ics.pbl.model.toResponse
import ac.osaka_u.ics.pbl.taskQueue
import io.ktor.http.*
import io.ktor.server.auth.*
import io.ktor.server.response.*
import java.util.*
import kotlinx.datetime.*

class AssignmentsHandler(private val assignmentRepo: AssignmentRepository, private val queueRepo: QueueRepository) {
    fun assignmentNext(id: String):  {
        val intId = id.toIntOrNull() ?: throw ApiException.BadRequestException("Invalid UserId")
        val assignedTask = assignmentRepo.findAssignmentByUserId(intId)
        if (assignedTask.isNotEmpty()){
            return assignedTask
        } else {
            val taskFromQueue = queueRepo.dequeue()
            if (taskFromQueue != null) {
                assignmentRepo.findAssignmentById(id) = taskFromQueue
                return taskFromQueue
                //call.respond(taskFromQueue)
            } else {
                val newTask = generateTask()
                if (newTask != null) {
                    taskQueue.add(newTask)
                    return newTask
                    //call.respond(newTask)
                } else {
                    return -1
                    //call.respond(HttpStatusCode.NoContent)
                }
            }
        }
        return taskRepo.findTasks().map { it.toResponse() }
    }

    fun convertTaskToAssignment(
        task: Task,
        clientId: Int,
        deadline: Instant = Clock.System.now().plus(5 * 60, kotlinx.datetime.DateTimeUnit.SECOND),
        status: AssignmentStatus = AssignmentStatus.PROCESSING
    ): Assignment {
        return Assignment(
            id = UUID.randomUUID(),
            assignedAt = Clock.System.now(),
            clientId = clientId,
            deadline = deadline,
            status = status,
            statusChangedAt = Clock.System.now(),
            task = task
        )
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
}