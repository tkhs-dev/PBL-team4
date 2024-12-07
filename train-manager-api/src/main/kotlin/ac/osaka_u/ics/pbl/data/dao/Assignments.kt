package ac.osaka_u.ics.pbl.data.dao

import ac.osaka_u.ics.pbl.common.AssignmentStatus
import org.jetbrains.exposed.dao.id.UUIDTable
import org.jetbrains.exposed.sql.kotlin.datetime.timestamp

object Assignments : UUIDTable("assignments") {
    val assignedAt = timestamp("assigned_at")
    val client = reference("client", Clients.id)
    val deadline = timestamp("deadline")
    val status = enumeration("status", AssignmentStatus::class)
    val statusChangedAt = timestamp("status_changed_at")
    val task = reference("task", Tasks.id)
}