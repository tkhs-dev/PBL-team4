package ac.osaka_u.ics.pbl.handler

import ac.osaka_u.ics.pbl.ApiException
import ac.osaka_u.ics.pbl.common.*
import ac.osaka_u.ics.pbl.domain.model.Model
import ac.osaka_u.ics.pbl.domain.model.Task
import ac.osaka_u.ics.pbl.domain.model.generateTask
import ac.osaka_u.ics.pbl.domain.repos.*
import ac.osaka_u.ics.pbl.model.*
import io.ktor.server.plugins.*
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant
import java.util.*
import kotlin.time.Duration.Companion.minutes

class AssignmentsHandler(private val assignmentRepos: AssignmentRepository, private val taskRepos:TaskRepository, private val errorRepos: ErrorRepository, private val queueRepository: QueueRepository, private val taskGeneratorRepository: TaskGeneratorRepository, private val modelRepository: ModelRepository) {
    private fun generateTask(): Task? {
        val generators = taskGeneratorRepository.findTaskGenerators().toMutableList()
        while (generators.isNotEmpty()) {
            val generator = generators.getRandomItemByWeight { it.weight.toDouble() } ?: return null
            val task = generator.generateTask(taskRepos, modelRepository)
            if (task != null) {
                return task
            }
            generators.remove(generator)
        }
        return null
    }

    fun handleGetNextAssignment(clientId: String): NextResponse? {
        val id = clientId.toIntOrNull() ?: throw BadRequestException("Invalid client id")
        val assignment = assignmentRepos.findProcessingAssignmentsByUserId(id)

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
                taskRepos.updateTask(it.task.id){
                    status = TaskStatus.WAITING
                }
                queueRepository.enqueue(it.task.id)
            }

        // キューからタスクを取得
        val task = queueRepository.dequeue() ?: generateTask() ?: return null
        taskRepos.updateTask(task.id){
            status = TaskStatus.PROCESSING
        }
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

    fun handleRegisterAssignment(id: String, clientId: String, request: AssignmentRegisterRequest, file: ByteArray): ModelResponse {
        val assignment = assignmentRepos.findAssignmentById(UUID.fromString(id)) ?: throw BadRequestException("Invalid assignment id")
        if (assignment.clientId != clientId.toIntOrNull()) {
            // クライアントIDが一致しない場合はエラー
            throw ApiException.ForbiddenException()
        }

        if (assignment.status != AssignmentStatus.PROCESSING) {
            // タスクが処理中でない場合はエラー
            throw BadRequestException("You can only register processing-state assignments")
        }

        val completedAt = Instant.fromEpochMilliseconds(request.completedAt)
        val modelId = UUID.randomUUID()

        val parentModel = assignment.task.baseModelId?.let {
            modelRepository.findModelById(it) ?: throw BadRequestException("Invalid model id")
        }

        modelRepository.saveModelFile(modelId, file)

        assignmentRepos.updateAssignment(assignment.id) {
            status = AssignmentStatus.COMPLETED
            statusChangedAt = completedAt
        }

        taskRepos.updateTask(assignment.task.id){
            status = TaskStatus.COMPLETED
        }

        val model =  modelRepository.createModel(
            Model(
                id = modelId,
                version = 1,
                parentId = parentModel?.id,
                rootModel = parentModel?.rootModel ?: modelId,
                sequence = parentModel?.sequence?.plus(1) ?: 0,
                taskId = assignment.task.id,
                createdAt = completedAt,
            )
        )

        return model.toResponse()
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

    fun handlePostError(id: String, clientId: String, request: AssignmentErrorRequest) {
        val assignment = assignmentRepos.findAssignmentById(UUID.fromString(id)) ?: throw BadRequestException("Invalid assignment id")
        if (assignment.clientId != clientId.toIntOrNull()) {
            // クライアントIDが一致しない場合はエラー
            throw ApiException.ForbiddenException()
        }

        if (assignment.status != AssignmentStatus.PROCESSING) {
            // タスクが処理中でない場合はエラー
            throw BadRequestException("You can only post error to processing-state assignments")
        }

        val timestamp = Clock.System.now()

        assignmentRepos.updateAssignment(assignment.id) {
            status = AssignmentStatus.ERROR
        }

        errorRepos.create(
            taskId = assignment.task.id,
            stackTrace = request.stackTrace,
            clientId = clientId.toInt(),
            assignmentId = assignment.id,
            timestamp = timestamp,
            clientVersion = request.clientVersion
        )

        if (assignment.task.errorCount + 1 <= 2) {
            queueRepository.enqueue(assignment.task.id)
            taskRepos.updateTask(assignment.task.id){
                status = TaskStatus.WAITING
                errorCount = task.errorCount + 1
            }
        }else{
            taskRepos.updateTask(assignment.task.id){
                status = TaskStatus.ERROR
                errorCount = task.errorCount + 1
            }
        }
    }
}