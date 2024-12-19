package ac.osaka_u.ics.pbl.model

import ac.osaka_u.ics.pbl.common.ParameterMapSerializer
import ac.osaka_u.ics.pbl.domain.model.Task
import ac.osaka_u.ics.pbl.domain.model.TaskGenerator
import kotlinx.serialization.Contextual
import kotlinx.serialization.Serializable

@Serializable
data class NextResponse(
    val id: String,
    val assigned_at: String,
    val client: String,
    val deadline: Long,
    val status: String,
    val status_changed_at: Long,
)

data class RefreshResponse(
    val id: String,
    val deadline: Long
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





