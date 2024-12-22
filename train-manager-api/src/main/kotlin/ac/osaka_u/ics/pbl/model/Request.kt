package ac.osaka_u.ics.pbl.model

import ac.osaka_u.ics.pbl.common.ParameterMapSerializer
import ac.osaka_u.ics.pbl.common.TaskType
import kotlinx.serialization.Contextual
import kotlinx.serialization.Serializable

@Serializable
data class AssignmentRegisterRequest(
    val completedAt: Long,
)

@Serializable
data class AssignmentErrorRequest(
    val stackTrace: String,
    val clientVersion: String
)

@Serializable
data class PostGeneratorRequest(
    val name: String,
    val type: TaskType,
    val weight: Int = 1,
    @Serializable(with = ParameterMapSerializer::class)
    val parameters: Map<String,@Contextual Any>
)

@Serializable
data class TaskRequest(
    val baseModelId: String?,
    val type: TaskType,
    @Serializable(with = ParameterMapSerializer::class)
    val parameters: Map<String,@Contextual Any>
)

@Serializable
data class ClientRequest(
    val user: String
)