package ac.osaka_u.ics.pbl.domain.model

import ac.osaka_u.ics.pbl.common.TaskType

data class TaskGenerator(
    val id: Int,
    val name: String,
    val type: TaskType,
    val weight: Int,
    val parameters: Map<String,Any>
)