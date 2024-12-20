package ac.osaka_u.ics.pbl.handler

import ac.osaka_u.ics.pbl.ApiException
import ac.osaka_u.ics.pbl.common.AssignmentStatus
import ac.osaka_u.ics.pbl.domain.model.Model
import ac.osaka_u.ics.pbl.domain.model.Task
import ac.osaka_u.ics.pbl.domain.repos.AssignmentRepository
import ac.osaka_u.ics.pbl.domain.repos.ModelRepository
import ac.osaka_u.ics.pbl.domain.repos.QueueRepository
import ac.osaka_u.ics.pbl.domain.repos.TaskGeneratorRepository
import ac.osaka_u.ics.pbl.model.ModelResponse
import ac.osaka_u.ics.pbl.model.NextResponse
import ac.osaka_u.ics.pbl.model.toNextResponse
import ac.osaka_u.ics.pbl.model.toResponse
import io.ktor.server.plugins.*
import kotlinx.datetime.Clock
import java.util.*
import kotlin.time.Duration.Companion.minutes

class AssignmentsHandler(private val assignmentRepos: AssignmentRepository, private val queueRepository: QueueRepository, private val taskGeneratorRepository: TaskGeneratorRepository, private val modelRepository: ModelRepository) {
    private fun generateTask(): Task? {
        return null
    }

    fun handleGetNextAssignment(clientId: String): NextResponse? {
        val id = clientId.toIntOrNull() ?: throw BadRequestException("Invalid client id")
        val assignment = assignmentRepos.findAssignmentByUserId(id)

        // 未処理のタスクがあればそれを返す
        assignment.firstOrNull{it.status == AssignmentStatus.PROCESSING}
            ?.let {
                // タスクがタイムアウトしていたらタイムアウト状態にする
                if (it.deadline < Clock.System.now()) {
                    assignmentRepos.updateAssignment(it.id){
                        status = AssignmentStatus.TIMEOUT
                    }
                }else{
                    return it.toNextResponse()
                }
            }

        assignmentRepos.findAssignmentsShouldBeTimeout()
            .forEach {
                assignmentRepos.updateAssignment(it.id){
                    status = AssignmentStatus.TIMEOUT
                }
                queueRepository.enqueue(it.task.id)
            }

        // キューからタスクを取得
        val task = queueRepository.dequeue() ?: generateTask() ?: return null

        // タスクを処理中にする
        return assignmentRepos.createAssignment(
            assignedAt = Clock.System.now(),
            clientId = id,
            deadline = Clock.System.now().plus(60.minutes),
            status = AssignmentStatus.PROCESSING,
            statusChangedAt = Clock.System.now(),
            taskId = task.id
        ).toNextResponse()
    }

    fun handleRegisterAssignment(id: String, clientId: String): ModelResponse {
        val assignment = assignmentRepos.findAssignmentById(UUID.fromString(id)) ?: throw BadRequestException("Invalid assignment id")
        if (assignment.clientId != clientId.toIntOrNull()) {
            // クライアントIDが一致しない場合はエラー
            throw ApiException.ForbiddenException()
        }

        if (assignment.status != AssignmentStatus.PROCESSING) {
            // タスクが処理中でない場合はエラー
            throw BadRequestException("You can only register processing-state assignments")
        }

        assignmentRepos.updateAssignment(assignment.id) {
            status = AssignmentStatus.COMPLETED
        }
        val parentModel = assignment.task.baseModelId?.let {
            modelRepository.findModelById(it) ?: throw BadRequestException("Invalid model id")
        }
        return modelRepository.createModel(
            Model(
                id = UUID.randomUUID(),
                version = 1,
                parentId = parentModel?.id,
                rootModel = parentModel?.rootModel,
                sequence = parentModel?.sequence?.plus(1) ?: 0,
                taskId = assignment.task.id,
                createdAt = Clock.System.now(),
            )
        ).toResponse()
    }

    fun handleRefreshAssignment(id: String, clientId: String) {
        val assignment = assignmentRepos.findAssignmentById(UUID.fromString(id)) ?: throw BadRequestException("Invalid assignment id")
        if (assignment.clientId != clientId.toIntOrNull()) {
            // クライアントIDが一致しない場合はエラー
            throw ApiException.ForbiddenException()
        }

        if (assignment.status != AssignmentStatus.PROCESSING) {
            // タスクが処理中でない場合はエラー
            throw BadRequestException("You can only refresh processing-state assignments")
        }

        assignmentRepos.updateAssignment(assignment.id) {
            deadline = Clock.System.now().plus(60.minutes)
            statusChangedAt = Clock.System.now()
        }
    }
}