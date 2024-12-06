package ac.osaka_u.ics.pbl.model

import kotlinx.serialization.Serializable

@Serializable
data class Task(
    val id: String,
    val completed: Boolean,
    val base_model_id: String,
    val type: String,
    val parameter: TaskParameter
)

interface TaskParameter

@Serializable
data class Parameter1(
    val player_id: String,
    val game_id: List<String>
) : TaskParameter