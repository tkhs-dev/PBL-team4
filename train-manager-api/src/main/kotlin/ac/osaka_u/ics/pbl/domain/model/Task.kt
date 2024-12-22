package ac.osaka_u.ics.pbl.domain.model

import ac.osaka_u.ics.pbl.common.TaskStatus
import ac.osaka_u.ics.pbl.common.TaskType
import kotlinx.datetime.Instant
import java.util.*

data class Task(
    val id: UUID,
    val status: TaskStatus,
    val errorCount: Int,
    val baseModelId: UUID?,
    val type: TaskType,
    val createdAt: Instant,
    val parameter: Map<String, Any>
)
