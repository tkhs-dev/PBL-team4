package ac.osaka_u.ics.pbl.model

import ac.osaka_u.ics.pbl.common.ParameterMapSerializer
import ac.osaka_u.ics.pbl.domain.model.Assignment
import ac.osaka_u.ics.pbl.domain.model.Model
import ac.osaka_u.ics.pbl.domain.model.Task
import ac.osaka_u.ics.pbl.domain.model.TaskGenerator
import kotlinx.serialization.Contextual
import kotlinx.serialization.Serializable

@Serializable
data class NextResponse(
    val id: String,
    val assigned_at: String,
    val deadline: Long,
    val task: TaskResponse
)

fun Assignment.toNextResponse() = NextResponse(
    id = id.toString(),
    assigned_at = assignedAt.toString(),
    deadline = deadline.toEpochMilliseconds(),
    task = task.toResponse()
)

@Serializable
data class RefreshResponse(
    val id: String,
    val deadline: Long
)

@Serializable
data class ModelResponse(
    val id: String,
    val version: Int,
    val parentId: String?,
    val rootId: String?,
    val sequence: Int,
    val taskId: String,
    val createdAt: Long,
)

fun Model.toResponse() = ModelResponse(
    id = id.toString(),
    version = version,
    parentId = parentId?.toString(),
    rootId = rootModel?.toString(),
    sequence = sequence,
    taskId = taskId.toString(),
    createdAt = createdAt.toEpochMilliseconds()
)

@Serializable
data class TaskResponse(
    val id: String,
    val completed: Boolean,
    val baseModelId: String?,
    val type: String,
    @Serializable(with = ParameterMapSerializer::class)
    val parameters: Map<String, @Contextual Any>
)

fun Task.toResponse() = TaskResponse(
    id = id.toString(),
    completed = completed,
    baseModelId = baseModelId?.toString(),
    type = type.name,
    parameters = parameter
)

@Serializable
data class TaskGeneratorResponse(
    val id: Int,
    val name: String,
    val type: String,
    val weight: Int,
    @Serializable(with = ParameterMapSerializer::class)
    val parameters: Map<String, @Contextual Any>
)

fun TaskGenerator.toResponse() = TaskGeneratorResponse(
    id = id,
    name = name,
    type = type.name,
    weight = weight,
    parameters = parameters
)




